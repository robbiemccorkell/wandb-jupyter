from typing import Literal
import uuid
import lightning
import torch
import numpy as np
import ray

from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.loggers import WandbLogger
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
)
import wandb

DEFAULT_WANDB_PROJECT = "hanging-runs-test"
LoggerType = Literal["ray", "lightning"]


# PyTorch Dataset for Tabular Data
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch Lightning DataModule for Iris Dataset
class IrisDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        iris = load_iris()
        self.data = iris.data
        self.target = iris.target
        self.num_classes = len(np.unique(self.target))

    def setup(self, stage=None):
        X_train, X_val, y_train, y_val = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Create PyTorch datasets
        self.train_dataset = TabularDataset(X_train, y_train)
        self.val_dataset = TabularDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# Fully Connected Neural Network Model
class FullyConnectedNet(lightning.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=64, learning_rate=1e-3):
        super(FullyConnectedNet, self).__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hparams.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.hidden_dim, output_dim),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long())
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long())
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Training function to be passed to Ray Tune
def train_tune(config, num_epochs, logger: LoggerType, wandb_project: str):
    # Set up data module with batch size from config
    iris_dm = IrisDataModule(batch_size=int(config["batch_size"]))
    iris_dm.prepare_data()
    iris_dm.setup()

    # Initialize model with hyperparameters from config
    model = FullyConnectedNet(
        input_dim=4,
        output_dim=3,
        hidden_dim=int(config["hidden_dim"]),
        learning_rate=config["learning_rate"],
    )

    # Initialize trainer
    trainer = lightning.Trainer(
        max_epochs=num_epochs,
        # Use Lightning WandbLogger
        # In combination with Ray Tune, this seems to leave runs in a 'running' state
        # after an HPO trial has ended
        logger=WandbLogger(project=wandb_project, log_model=False)
        if logger == "lightning"
        else None,
        enable_progress_bar=False,
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
    )

    # Train the model
    trainer.fit(model, iris_dm)


def hpo(
    num_samples=10,
    num_epochs=1000,
    logger: LoggerType = "lightning",
    wandb_project=DEFAULT_WANDB_PROJECT,
):
    # Generate a unique id for this run
    run_id = str(uuid.uuid4()).split("-")[0]

    # Start a wandb log for the whole training run
    wandb.init(
        project=wandb_project,
        name=f"{run_id}-train",
        config={
            "num_samples": num_samples,
            "num_epochs": num_epochs,
            "logger": logger,
        },
        tags=["train"],
    )

    # Define the search space
    search_space = {
        "hidden_dim": ray.tune.choice([16, 32, 64, 128]),
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        "batch_size": ray.tune.choice([16, 32, 64]),
    }

    # Run the Ray Tune experiment
    tuner = ray.tune.Tuner(
        ray.tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            logger=logger,
            wandb_project=wandb_project,
        ),
        param_space=search_space,
        tune_config=ray.tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
        ),
        # This logger correctly ends the run after an HPO trial has ended
        run_config=ray.train.RunConfig(
            callbacks=[WandbLoggerCallback(project=wandb_project)]
        )
        if logger == "ray"
        else None,
    )
    results = tuner.fit()

    # Get the best trial
    best_result = results.get_best_result("val_loss", "min")
    print("Best hyperparameters found were: ", best_result.config)

    wandb.log({"train_metrics": best_result.metrics})
    wandb.finish()

    return best_result
