import os
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from model import CrowClassifier  # Your LightningModule
from dataset import CrowDataset  # Your Dataset


def evaluate_model(model, dataloader, logger, device):
    model.eval()
    # Create dictionaries to accumulate correct counts and totals per attribute.
    metrics = {
        "crowCount": {"correct": 0, "total": 0},
        "crowAge": {"correct": 0, "total": 0},
        "begging": {"correct": 0, "total": 0},
        "softSong": {"correct": 0, "total": 0},
        "rattle": {"correct": 0, "total": 0},
        "badQuality": {"correct": 0, "total": 0},
        "human": {"correct": 0, "total": 0}
    }

    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            # Move all labels to the same device.
            for key in labels:
                labels[key] = labels[key].to(device)

            outputs = model(embeddings)
            # For multi-class attributes, use argmax.
            pred_crowCount = torch.argmax(outputs["crowCount"], dim=1)
            pred_crowAge = torch.argmax(outputs["crowAge"], dim=1)
            # For binary attributes, threshold the logits at 0.
            pred_begging = (outputs["begging"].squeeze() > 0).long()
            pred_softSong = (outputs["softSong"].squeeze() > 0).long()
            pred_rattle = (outputs["rattle"].squeeze() > 0).long()
            pred_badQuality = (outputs["badQuality"].squeeze() > 0).long()
            pred_human = (outputs["human"].squeeze() > 0).long()

            # Accumulate counts.
            metrics["crowCount"]["correct"] += (pred_crowCount == labels["crowCount"]).sum().item()
            metrics["crowCount"]["total"] += labels["crowCount"].size(0)

            metrics["crowAge"]["correct"] += (pred_crowAge == labels["crowAge"]).sum().item()
            metrics["crowAge"]["total"] += labels["crowAge"].size(0)

            metrics["begging"]["correct"] += (pred_begging == labels["begging"]).sum().item()
            metrics["begging"]["total"] += labels["begging"].size(0)

            metrics["softSong"]["correct"] += (pred_softSong == labels["softSong"]).sum().item()
            metrics["softSong"]["total"] += labels["softSong"].size(0)

            metrics["rattle"]["correct"] += (pred_rattle == labels["rattle"]).sum().item()
            metrics["rattle"]["total"] += labels["rattle"].size(0)

            metrics["badQuality"]["correct"] += (pred_badQuality == labels["badQuality"]).sum().item()
            metrics["badQuality"]["total"] += labels["badQuality"].size(0)

            metrics["human"]["correct"] += (pred_human == labels["human"]).sum().item()
            metrics["human"]["total"] += labels["human"].size(0)

    # Calculate accuracies.
    accuracies = {}
    for key, counts in metrics.items():
        accuracies[key] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0

    # Log each metric to TensorBoard.
    global_step = 0  # Adjust if needed (e.g., current epoch or iteration).
    for key, acc in accuracies.items():
        logger.experiment.add_scalar(f"accuracy/{key}", acc, global_step)
        print(f"Accuracy for {key}: {acc:.4f}")

    return accuracies


if __name__ == "__main__":
    # Set device (GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and a DataLoader for evaluation.
    # You could use your validation split here if preferred.
    dataset = CrowDataset()

    # Determine sizes for training and validation sets.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=3, shuffle=False)

    # Instantiate the model.
    model = CrowClassifier()
    # Optionally, load a checkpoint:
    model = CrowClassifier.load_from_checkpoint("logs/checkpoints/best_model.ckpt")
    model.to(device)

    # Setup TensorBoard logger.
    tb_logger = TensorBoardLogger("logs", name="crow-classify")

    # Run evaluation.
    accuracies = evaluate_model(model, train_loader, tb_logger, device)
