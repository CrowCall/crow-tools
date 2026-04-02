import argparse
import os

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import DATASET_SPLIT_SEED, CrowDataset, split_train_val_dataset
import evaluate as evaluate_module
from model import CrowClassifier

CLASSIFIER_DIR = os.path.dirname(__file__)
LOGS_DIR = os.path.join(CLASSIFIER_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(LOGS_DIR, "checkpoints")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the classifier on a dataset.")
    parser.add_argument("--dataset", default="all-public", help="Dataset to train on.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowClassifier()
    model.to(device)

    dataset = CrowDataset(dataset_name=args.dataset)
    train_dataset, val_dataset = split_train_val_dataset(dataset)

    oversample_factors = {"rattle": 4, "softSong": 4, "begging": 1, "alert": 3, "mob": 1}
    oversampled_train_indices = []
    for idx in train_dataset.indices:
        _, label = dataset[idx]
        factor = 1
        for flag, dup_factor in oversample_factors.items():
            if label.get(flag, 0) == 1:
                factor *= dup_factor
        oversampled_train_indices.extend([idx] * factor)

    oversampled_train_dataset = Subset(dataset, oversampled_train_indices)

    train_loader = DataLoader(oversampled_train_dataset, batch_size=21, num_workers=3, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=21, num_workers=3, shuffle=False, drop_last=True)

    tb_logger = TensorBoardLogger(LOGS_DIR, name=f"crow-classify-{args.dataset}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename=f"best_model-{args.dataset}",
        monitor="val_composite_score",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(max_epochs=20, logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_checkpoint = os.path.abspath(best_checkpoint)
    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Validation split seed: {DATASET_SPLIT_SEED}")
    print("Running evaluation with the best checkpoint from this training run...")
    eval_argv = [
        "--dataset",
        args.dataset,
        "--checkpoint",
        best_checkpoint,
    ]
    evaluate_module.main(eval_argv)


if __name__ == "__main__":
    main()
