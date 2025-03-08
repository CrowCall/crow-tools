from torch import optim
from asteroid.models import DPRNNTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.engine import System
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from separator.dataset import *

json_path = "labeler-vue/public/mixes/mix-dataset.json"
merged_dir = "labeler-vue/public/mixes/merged"
separate_dir = "labeler-vue/public/mixes/separate"
sr = 16000

train_dataset = CrowMixDataset(json_path, merged_dir, separate_dir, sr=sr)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Optionally, create a validation dataset similarly.
val_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Specify number of sources (e.g., 2 if you mix background and one crow call,
# or adjust based on your scenario)
model = DPRNNTasNet(n_src=FIXED_N_SRC)

loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create the training system with your custom DataLoader(s)
tb_logger = TensorBoardLogger("logs", name="asteroid-crow")
system = System(model, optimizer, loss, train_loader, val_loader)

trainer = Trainer(max_epochs=50, logger=tb_logger, log_every_n_steps=1)
trainer.fit(system)
