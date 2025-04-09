import optuna
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import CrowDataset
from model import CrowClassifier


def objective(trial: optuna.Trial):
    # Sample hyperparameters.
    hidden_dim = trial.suggest_int("hidden_dim", 237, 237)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.3)
    random_seed = trial.suggest_int("random_seed", 0, 20000)
    learning_rate = trial.suggest_float("learning_rate", 0.000145, 0.000145)
    batch_size = trial.suggest_int("batch_size", 18, 24)
    rattle_oversample = trial.suggest_int("rattle_oversample", 1, 4)
    softsong_oversample = trial.suggest_int("softsong_oversample", 1, 4)
    begging_oversample = trial.suggest_int("begging_oversample", 1, 4)
    alert_oversample = trial.suggest_int("alert_oversample", 1, 4)
    mob_oversample = trial.suggest_int("mob_oversample", 1, 4)

    # Instantiate the model with sampled hyperparameters.
    model = CrowClassifier(hidden_dim=hidden_dim, dropout_rate=dropout_rate, seed=random_seed, lr=learning_rate)

    # Create the dataset and split into train/validation.
    dataset = CrowDataset()
    train_size = int(0.88 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Oversampling: Duplicate training indices for underrepresented labels.
    oversample_factors = {"rattle": rattle_oversample, "softSong": softsong_oversample, "begging": begging_oversample, "alert": alert_oversample, "mob": mob_oversample}
    oversampled_train_indices = []
    for idx in train_dataset.indices:
        _, label = dataset[idx]
        factor = 1
        for flag, dup_factor in oversample_factors.items():
            if label.get(flag, 0) == 1:
                factor *= dup_factor
        oversampled_train_indices.extend([idx] * factor)

    oversampled_train_dataset = Subset(dataset, oversampled_train_indices)

    train_loader = DataLoader(oversampled_train_dataset, batch_size=batch_size, num_workers=3, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=True)

    # Set up checkpoint callback to monitor the composite score.
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/trial_{trial.number}/checkpoints",
        filename="best_model",
        monitor="val_composite_score",
        mode="max",  # We want to maximize composite score.
        save_top_k=1,
    )

    tb_logger = TensorBoardLogger("logs", name=f"crow-classify_trial_{trial.number}")

    trainer = pl.Trainer(
        max_epochs=12,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Retrieve the best composite score.
    best_val_score = checkpoint_callback.best_model_score
    if best_val_score is None:
        # Fall back: try to get the last logged composite score.
        best_val_score = trainer.callback_metrics.get("val_composite_score")
        if best_val_score is None:
            raise ValueError("Validation composite score not found!")

    # Since we are maximizing the composite score, the higher the better.
    return best_val_score.item() if isinstance(best_val_score, torch.Tensor) else best_val_score


if __name__ == "__main__":
    # Change study direction to "maximize" because higher composite score is better.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
