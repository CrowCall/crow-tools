import optuna
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import CrowDataset
from model import CrowClassifier


def objective(trial: optuna.Trial):
    # Sample hyperparameters.
    hidden_dim = trial.suggest_int("hidden_dim", 200, 280)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    random_seed = trial.suggest_int("random_seed", 0, 9000)
    learning_rate = trial.suggest_float("learning_rate", 0.0006, 0.0008)
    batch_size = trial.suggest_int("batch_size", 18, 24)

    # Instantiate the model with sampled hyperparameters.
    model = CrowClassifier(hidden_dim=hidden_dim, dropout_rate=dropout_rate, seed=random_seed, lr=learning_rate)

    # Create the dataset and split into train/validation.
    dataset = CrowDataset()
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=True)

    # Set up checkpoint callback (store checkpoints in a trial-specific directory).
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/trial_{trial.number}/checkpoints",
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Create a TensorBoard logger (also trial-specific).
    tb_logger = TensorBoardLogger("logs", name=f"crow-classify_trial_{trial.number}")

    # Initialize the Trainer.
    trainer = pl.Trainer(
        max_epochs=11,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,  # disable progress bar for cleaner logs during tuning
    )

    # Train the model.
    trainer.fit(model, train_loader, val_loader)

    # Retrieve the best validation loss.
    best_val_loss = checkpoint_callback.best_model_score
    if best_val_loss is None:
        # Fall back to the last epoch's metric if checkpoint callback didn't update.
        best_val_loss = trainer.callback_metrics.get("val_loss")
        if best_val_loss is None:
            raise ValueError("Validation loss not found!")

    # Return the best validation loss (make sure it is a Python float).
    return best_val_loss.item() if isinstance(best_val_loss, torch.Tensor) else best_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
