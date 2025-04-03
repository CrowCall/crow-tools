import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split, Subset
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

# Split dataset into training and validation (ensuring unique records in each).
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Oversampling: Duplicate training indices for underrepresented labels.
oversample_factors = {"rattle": 2, "softSong": 3, "begging": 3, "alert": 1, "mob": 1}

oversampled_train_indices = []
# train_dataset.indices gives the list of indices from the original dataset in the training subset.
for idx in train_dataset.indices:
    _, label = dataset[idx]
    factor = 1
    # Multiply factors for each applicable flag.
    for flag, dup_factor in oversample_factors.items():
        if label.get(flag, 0) == 1:
            factor *= dup_factor
    # Add this index 'factor' times.
    oversampled_train_indices.extend([idx] * factor)

# Create a new training subset with the oversampled indices.
oversampled_train_dataset = Subset(dataset, oversampled_train_indices)

# Create DataLoaders.
train_loader = DataLoader(oversampled_train_dataset, batch_size=23, num_workers=3, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=23, num_workers=3, shuffle=False, drop_last=True)

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
