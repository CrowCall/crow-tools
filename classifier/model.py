import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning_fabric import seed_everything

# Seed for reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CrowClassifier(pl.LightningModule):
    def __init__(self, input_dim=768, hidden_dim=244, dropout_rate=0.26704036009338883, seed=557, lr=0.0007331937958897578):
        super().__init__()

        seed_everything(seed, workers=True)
        self.lr = lr

        # A simple shared backbone.
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Output heads for each task.
        self.crowCount_head = nn.Linear(hidden_dim, 5)  # 5 classes (labels: 0,1,2,3,4)
        self.crowAge_head = nn.Linear(hidden_dim, 2)    # 2 classes: adult vs juvenile
        self.alert_head = nn.Linear(hidden_dim, 1)      # binary
        self.begging_head = nn.Linear(hidden_dim, 1)    # binary
        self.softSong_head = nn.Linear(hidden_dim, 1)   # binary
        self.rattle_head = nn.Linear(hidden_dim, 1)     # binary
        self.mob_head = nn.Linear(hidden_dim, 1)        # binary
        self.quality_head = nn.Linear(hidden_dim, 3)    # 3 classes: bad, average, HQ

        # Loss functions.
        self.loss_fn_class = nn.CrossEntropyLoss()
        self.loss_fn_bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        rep = self.backbone(x)
        out = {
            "crowCount": self.crowCount_head(rep),
            "crowAge": self.crowAge_head(rep),
            "alert": self.alert_head(rep),
            "begging": self.begging_head(rep),
            "softSong": self.softSong_head(rep),
            "rattle": self.rattle_head(rep),
            "mob": self.mob_head(rep),
            "quality": self.quality_head(rep)
        }
        return out

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = (
                1.0 * self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
                1.0 * self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
                1.0 * self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
                1.0 * self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
                2.0 * self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
                2.0 * self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
                1.0 * self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
                1.0 * self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = (
                1.0 * self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
                1.0 * self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
                1.0 * self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
                1.0 * self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
                2.0 * self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
                2.0 * self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
                1.0 * self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
                1.0 * self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)