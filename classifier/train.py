import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import CrowDataset
from model import CrowClassifier

# Instantiate the model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowClassifier()
model.to(device)

# Create the dataset.
dataset = CrowDataset()

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=21, num_workers=3, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=21, num_workers=3, shuffle=False, drop_last=True)

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
trainer = pl.Trainer(max_epochs=20, logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
