import torch
from torch.utils.data import DataLoader, random_split, Subset
from dataset import CrowDataset
from model import CrowClassifier
import os

PATH = os.path.dirname(__file__)
checkpoint_path = os.path.join(PATH, "logs", "checkpoints", "best_model.ckpt")

def print_overall_metrics(metrics):
    print("\n=== Overall Accuracy ===")
    for key, counts in metrics.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        print(f"{key:10s}: {acc * 100:6.2f}% ({counts['correct']}/{counts['total']})")


def print_breakdown(title, breakdown, label_prefix="Class"):
    print(f"\n=== {title} Breakdown ===")
    for cls, (correct, total) in breakdown.items():
        acc = correct / total if total > 0 else 0.0
        print(f"  {label_prefix} {cls}: {acc * 100:6.2f}% ({correct}/{total})")


def evaluate_model(model, dataloader, device):
    model.eval()
    # Overall metrics for each attribute.
    metrics = {
        "crowCount": {"correct": 0, "total": 0},
        "crowAge": {"correct": 0, "total": 0},
        "quality": {"correct": 0, "total": 0},
        "alert": {"correct": 0, "total": 0},
        "begging": {"correct": 0, "total": 0},
        "softSong": {"correct": 0, "total": 0},
        "rattle": {"correct": 0, "total": 0},
        "mob": {"correct": 0, "total": 0}
    }

    # For per-class breakdown.
    # For crowCount, we now have 5 classes (0-4), while crowAge and quality remain 2 and 3 classes (1-indexed)
    multi_class_keys = {"crowCount": 5, "crowAge": 2, "quality": 2}
    breakdown = {}
    for key, num_classes in multi_class_keys.items():
        if key == "crowCount":
            breakdown[key] = {cls: [0, 0] for cls in range(num_classes)}
        else:
            breakdown[key] = {cls + 1: [0, 0] for cls in range(num_classes)}

    # For binary attributes breakdown.
    binary_keys = ["alert", "begging", "softSong", "rattle", "mob"]
    breakdown_binary = {key: {0: [0, 0], 1: [0, 0]} for key in binary_keys}

    with torch.no_grad():
        for batch in dataloader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            for key in labels:
                labels[key] = labels[key].to(device)

            outputs = model(embeddings)
            # Multi-class predictions (0-indexed).
            pred_crowCount = torch.argmax(outputs["crowCount"], dim=1)
            pred_crowAge = torch.argmax(outputs["crowAge"], dim=1)
            pred_quality = torch.argmax(outputs["quality"], dim=1)
            # Binary predictions: threshold at 0 and force 1D.
            pred_alert = (outputs["alert"].squeeze() > 0).long().view(-1)
            pred_begging = (outputs["begging"].squeeze() > 0).long().view(-1)
            pred_softSong = (outputs["softSong"].squeeze() > 0).long().view(-1)
            pred_rattle = (outputs["rattle"].squeeze() > 0).long().view(-1)
            pred_mob = (outputs["mob"].squeeze() > 0).long().view(-1)

            # Update overall metrics (for multi-class, crowCount no longer subtracts 1).
            for key, pred, gt in [
                ("crowCount", pred_crowCount, labels["crowCount"]),
                ("crowAge", pred_crowAge, labels["crowAge"] - 1),
                ("quality", pred_quality, labels["quality"] - 1)
            ]:
                metrics[key]["correct"] += (pred == gt).sum().item()
                metrics[key]["total"] += gt.size(0)
                # Per-class breakdown.
                for cls in range(multi_class_keys[key]):
                    mask = (gt == cls) if key == "crowCount" else (gt == cls)
                    total = mask.sum().item()
                    correct = ((pred == cls) & mask).sum().item()
                    if key == "crowCount":
                        breakdown[key][cls][0] += correct
                        breakdown[key][cls][1] += total
                    else:
                        # For crowAge and quality, we display classes as 1-indexed.
                        breakdown[key][cls + 1][0] += correct
                        breakdown[key][cls + 1][1] += total

            # Update overall metrics for binary attributes.
            for key, pred in [
                ("alert", pred_alert),
                ("begging", pred_begging),
                ("softSong", pred_softSong),
                ("rattle", pred_rattle),
                ("mob", pred_mob)
            ]:
                metrics[key]["correct"] += (pred == labels[key]).sum().item()
                metrics[key]["total"] += labels[key].size(0)
                # Breakdown per value.
                for val in [0, 1]:
                    mask = (labels[key] == val)
                    total = mask.sum().item()
                    correct = ((pred == val) & mask).sum().item()
                    breakdown_binary[key][val][0] += correct
                    breakdown_binary[key][val][1] += total

    print_overall_metrics(metrics)

    for key in multi_class_keys:
        print_breakdown(key, breakdown[key], label_prefix="Class")

    for key in binary_keys:
        print_breakdown(key, breakdown_binary[key], label_prefix="Value")

    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrowClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)

    # Create the dataset.
    dataset = CrowDataset()

    train_size = int(0.88 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders.
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=3, shuffle=False)

    print(f"\nEvaluating on {len(val_loader)} VALIDATE samples")
    evaluate_model(model, val_loader, device)
