from torch import optim
from asteroid.models import DPRNNTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.engine import System
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
from dataset import *

# Seed for reproducibility
seed_everything(42, workers=True)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Enable Tensor Core optimization
torch.set_float32_matmul_precision('high')

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

json_path = os.path.join("../.cache", "mixes", "mix-dataset.json")
merged_dir = os.path.join("../.cache", "mixes", "merged")
separate_dir = os.path.join("../.cache", "mixes", "separate")
sr = 8000
batch_size = 1

# Load dataset
dataset = CrowMixDataset(json_path, merged_dir, separate_dir, sr=sr)

# Determine sizes for training and validation sets.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Get data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, persistent_workers=True, collate_fn=collate_fn)

# Specify number of sources (e.g., 2 if you mix background and one crow call)
model = DPRNNTasNet(n_src=FIXED_N_SRC).to(device)

loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Create the training system with your custom DataLoader(s)
tb_logger = TensorBoardLogger("logs", name="crow")
system = System(model, optimizer, loss, train_loader, val_loader)

# Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="logs/crow/checkpoints",
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=3
)

# Move Trainer to use CPU instead of GPU
trainer = Trainer(
    max_epochs=50,
    logger=tb_logger,
    log_every_n_steps=1,
    accumulate_grad_batches=2,
    precision="16-mixed",
    #accelerator="gpu",
    deterministic=True,
    callbacks=[checkpoint_callback]
)

trainer.fit(system)
