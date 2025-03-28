from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import CrowDataset
from model import CrowClassifier
import random

# Create the dataset.
dataset = CrowDataset()

# Create a list of indices and shuffle them.
indices = list(range(len(dataset)))
random.shuffle(indices)

# Compute the split index for 85% training and 15% validation.
split = int(0.85 * len(dataset))
train_indices = indices[:split]
val_indices = indices[split:]

# Create Subset datasets.
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=14, num_workers=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=14, num_workers=3, shuffle=False)

# Instantiate the model.
model = CrowClassifier()

# Create TensorBoard logger.
tb_logger = TensorBoardLogger("logs", name="crow-classify")

# Checkpointing.
checkpoint_callback = ModelCheckpoint(
    dirpath="logs/checkpoints",
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=3
)

# Initialize the Trainer.
trainer = pl.Trainer(max_epochs=60, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
