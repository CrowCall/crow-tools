import torch
import torch.nn as nn
import pytorch_lightning as pl

seed = 42
pl.seed_everything(seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CrowClassifier(pl.LightningModule):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        # A simple shared backbone.
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate output heads:
        self.crowCount_head = nn.Linear(hidden_dim, 2)  # e.g. "single" vs "multiple"
        self.crowAge_head = nn.Linear(hidden_dim, 2)  # "adult" vs "juvenile"
        self.begging_head = nn.Linear(hidden_dim, 1)
        self.softSong_head = nn.Linear(hidden_dim, 1)
        self.rattle_head = nn.Linear(hidden_dim, 1)
        self.badQuality_head = nn.Linear(hidden_dim, 1)
        self.human_head = nn.Linear(hidden_dim, 1)

        # Define loss functions.
        self.loss_fn_class = nn.CrossEntropyLoss()
        self.loss_fn_bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.backbone(x)
        out = {
            "crowCount": self.crowCount_head(x),
            "crowAge": self.crowAge_head(x),
            "begging": self.begging_head(x),
            "softSong": self.softSong_head(x),
            "rattle": self.rattle_head(x),
            "badQuality": self.badQuality_head(x),
            "human": self.human_head(x)
        }
        return out

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)

        # Calculate the losses for each task.
        loss = self.loss_fn_class(outputs["crowCount"], labels["crowCount"]) + \
               self.loss_fn_class(outputs["crowAge"], labels["crowAge"]) + \
               self.loss_fn_bce(outputs["begging"].squeeze(), labels["begging"].float()) + \
               self.loss_fn_bce(outputs["softSong"].squeeze(), labels["softSong"].float()) + \
               self.loss_fn_bce(outputs["rattle"].squeeze(), labels["rattle"].float()) + \
               self.loss_fn_bce(outputs["badQuality"].squeeze(), labels["badQuality"].float()) + \
               self.loss_fn_bce(outputs["human"].squeeze(), labels["human"].float())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)

        loss = self.loss_fn_class(outputs["crowCount"], labels["crowCount"]) + \
               self.loss_fn_class(outputs["crowAge"], labels["crowAge"]) + \
               self.loss_fn_bce(outputs["begging"].squeeze(), labels["begging"].float()) + \
               self.loss_fn_bce(outputs["softSong"].squeeze(), labels["softSong"].float()) + \
               self.loss_fn_bce(outputs["rattle"].squeeze(), labels["rattle"].float()) + \
               self.loss_fn_bce(outputs["badQuality"].squeeze(), labels["badQuality"].float()) + \
               self.loss_fn_bce(outputs["human"].squeeze(), labels["human"].float())

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)