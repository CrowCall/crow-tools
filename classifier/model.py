import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning_fabric import seed_everything

# Seed for reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CrowClassifier(pl.LightningModule):
    def __init__(self, input_dim=768, hidden_dim=237, dropout_rate=0.3, seed=7204, lr=0.000145):
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
        self.quality_head = nn.Linear(hidden_dim, 2)    # 3 classes: bad, average, HQ

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
            self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
            self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
            self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
            self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
            self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
            self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
            self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
            self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute predictions and loss on the validation batch.
        Returns a dict of predictions, ground truths, and loss.
        """
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = (
            self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
            self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
            self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
            self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
            self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
            self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
            self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
            self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )

        # Multi-class predictions (0-indexed).
        pred_crowCount = torch.argmax(outputs["crowCount"], dim=1)
        pred_crowAge = torch.argmax(outputs["crowAge"], dim=1)
        pred_quality = torch.argmax(outputs["quality"], dim=1)
        # Binary predictions.
        pred_alert = (outputs["alert"].squeeze() > 0).long().view(-1)
        pred_begging = (outputs["begging"].squeeze() > 0).long().view(-1)
        pred_softSong = (outputs["softSong"].squeeze() > 0).long().view(-1)
        pred_rattle = (outputs["rattle"].squeeze() > 0).long().view(-1)
        pred_mob = (outputs["mob"].squeeze() > 0).long().view(-1)

        out = {
            "crowCount": {"pred": pred_crowCount, "gt": labels["crowCount"]},
            "crowAge": {"pred": pred_crowAge, "gt": labels["crowAge"] - 1},  # adjust as before
            "quality": {"pred": pred_quality, "gt": labels["quality"] - 1},
            "alert": {"pred": pred_alert, "gt": labels["alert"]},
            "begging": {"pred": pred_begging, "gt": labels["begging"]},
            "softSong": {"pred": pred_softSong, "gt": labels["softSong"]},
            "rattle": {"pred": pred_rattle, "gt": labels["rattle"]},
            "mob": {"pred": pred_mob, "gt": labels["mob"]},
            "loss": loss
        }
        # Append output to a temporary list (see on_validation_epoch_* hooks below).
        self._validation_outputs.append(out)
        return out

    def on_validation_epoch_start(self):
        # Clear the validation outputs at the start of the epoch.
        self._validation_outputs = []

    def on_validation_epoch_end(self):
        """
        Aggregate all validation_step outputs, compute per-task breakdowns,
        calculate the composite metric, and log the results.
        """
        # Expected number of classes for multi-class tasks.
        multi_class_keys = {"crowCount": 5, "crowAge": 2, "quality": 2}
        breakdown = {key: {cls: [0, 0] for cls in range(num)} for key, num in multi_class_keys.items()}
        # Binary tasks.
        binary_keys = ["alert", "begging", "softSong", "rattle", "mob"]
        breakdown_binary = {key: {0: [0, 0], 1: [0, 0]} for key in binary_keys}

        total_loss = 0.0
        batch_count = len(self._validation_outputs)
        for batch_out in self._validation_outputs:
            total_loss += batch_out["loss"].item()
            # Multi-class tasks.
            for task in multi_class_keys:
                preds = batch_out[task]["pred"].detach().cpu()
                gts = batch_out[task]["gt"].detach().cpu()
                for cls in range(multi_class_keys[task]):
                    mask = (gts == cls)
                    total = mask.sum().item()
                    correct = ((preds == cls) & mask).sum().item()
                    breakdown[task][cls][0] += correct
                    breakdown[task][cls][1] += total
            # Binary tasks.
            for task in binary_keys:
                preds = batch_out[task]["pred"].detach().cpu()
                gts = batch_out[task]["gt"].detach().cpu()
                for val in [0, 1]:
                    mask = (gts == val)
                    total = mask.sum().item()
                    correct = ((preds == val) & mask).sum().item()
                    breakdown_binary[task][val][0] += correct
                    breakdown_binary[task][val][1] += total

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        composite, task_scores = self.compute_composite_score(breakdown, breakdown_binary)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_composite_score", composite, prog_bar=True)
        for task, score in task_scores.items():
            self.log(f"val_{task}_score", score)
        # (Optionally, you can also print or store these metrics.)

    def compute_composite_score(self, breakdown, breakdown_binary, weights=None):
        """
        Compute a composite score as a weighted average of:
          - Macro accuracy for multi-class tasks (crowCount, crowAge, quality)
          - Balanced accuracy for binary tasks (alert, begging, softSong, rattle, mob)
        Args:
            breakdown (dict): Breakdown dict for multi-class tasks.
            breakdown_binary (dict): Breakdown dict for binary tasks.
            weights (dict, optional): Task weights.
              Defaults to:
              {
                  "crowCount": 2,
                  "crowAge": 1,
                  "quality": 1,
                  "alert": 1.5,
                  "begging": 1.5,
                  "softSong": 1.5,
                  "rattle": 2,
                  "mob": 2
              }
        Returns:
            composite_score (float): Weighted composite score.
            task_scores (dict): Dictionary of individual task scores.
        """
        if weights is None:
            weights = {
                "crowCount": 2,
                "crowAge": 1,
                "quality": 1,
                "alert": 1.5,
                "begging": 1.5,
                "softSong": 1.5,
                "rattle": 2,
                "mob": 2
            }
        task_scores = {}

        # Multi-class tasks: macro accuracy = average accuracy over all classes.
        multi_class_tasks = ["crowCount", "crowAge", "quality"]
        for task in multi_class_tasks:
            class_accuracies = []
            for cls, (correct, total) in breakdown[task].items():
                if total > 0:
                    class_accuracies.append(correct / total)
            task_scores[task] = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

        # Binary tasks: balanced accuracy = (acc_0 + acc_1) / 2.
        binary_tasks = ["alert", "begging", "softSong", "rattle", "mob"]
        for task in binary_tasks:
            acc_vals = []
            for val in [0, 1]:
                correct, total = breakdown_binary[task][val]
                acc_vals.append((correct / total) if total > 0 else 0.0)
            task_scores[task] = sum(acc_vals) / 2.0

        # Compute weighted average.
        total_weighted = 0.0
        total_weights = 0.0
        for task, score in task_scores.items():
            weight = weights.get(task, 1)
            total_weighted += weight * score
            total_weights += weight
        composite_score = total_weighted / total_weights if total_weights > 0 else 0.0
        return composite_score, task_scores

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)