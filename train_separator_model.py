from torch import optim
from asteroid.models import DPRNNTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.engine import System
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from separator.dataset import *
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available
print(torch.cuda.device_count())  # Should print number of GPUs available
print(torch.cuda.get_device_name(0))  # Prints GPU model

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

json_path = "labeler-vue/public/mixes/mix-dataset.json"
merged_dir = "labeler-vue/public/mixes/merged"
separate_dir = "labeler-vue/public/mixes/separate"
sr = 16000
batch_size = 10 #5

train_dataset = CrowMixDataset(json_path, merged_dir, separate_dir, sr=sr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True, collate_fn=collate_fn)

# Optionally, create a validation dataset similarly.
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=5, persistent_workers=True, collate_fn=collate_fn)

#from asteroid.data import LibriMix
#train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_noisy", batch_size=16)
#train_loader.dataset.df["length"] = train_loader.dataset.df["length"].astype(int)
#val_loader.dataset.df["length"] = val_loader.dataset.df["length"].astype(int)


# Specify number of sources (e.g., 2 if you mix background and one crow call)
model = DPRNNTasNet(n_src=FIXED_N_SRC).to(device)

loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Create the training system with your custom DataLoader(s)
tb_logger = TensorBoardLogger("logs", name="asteroid-crow")
system = System(model, optimizer, loss, train_loader, val_loader)

# Move Trainer to use CPU instead of GPU
trainer = Trainer(
    max_epochs=50,
    logger=tb_logger,
    log_every_n_steps=1,
    accumulate_grad_batches=3,
    precision="16-mixed",
    accelerator="gpu"
)

trainer.fit(system)
