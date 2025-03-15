from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import CrowClassifier
from dataset import CrowDataset

# Create the dataset.
dataset = CrowDataset()

# Determine sizes for training and validation sets.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for each split.
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=3, shuffle=False)

# Instantiate the model.
model = CrowClassifier()

# Create TensorBoard logger.
tb_logger = TensorBoardLogger("logs", name="crow-classify")

# Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="logs/checkpoints",
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=3
)

# Initialize the Trainer with the logger and train.
trainer = pl.Trainer(max_epochs=60, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
