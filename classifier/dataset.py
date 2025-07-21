import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

PATH = os.path.dirname(__file__)
CACHE_BASE = os.path.join(PATH, "..", ".cache")
LIBRARIES_BASE = os.path.join(CACHE_BASE, "libraries")

def locate_embedding(file_id, denoised=True):
    """Return path to the embedding for ``file_id`` in any library."""
    suffix = "embeddings-denoised" if denoised else "embeddings"
    for lib in os.listdir(LIBRARIES_BASE):
        emb_dir = os.path.join(LIBRARIES_BASE, lib, suffix)
        path = os.path.join(emb_dir, f"{file_id}.npy")
        if os.path.exists(path):
            return path
    return None

class CrowDataset(Dataset):
    def __init__(self, dataset_name="all-public"):
        dataset_dir = os.path.join(CACHE_BASE, "datasets", dataset_name)
        labels_file = os.path.join(dataset_dir, "labels.json")
        config_file = os.path.join(dataset_dir, "config.json")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels not found for dataset {dataset_name}")

        self.included_libraries = []
        if os.path.exists(config_file):
            with open(config_file, 'r') as cf:
                cfg = json.load(cf)
                self.included_libraries = cfg.get("included_libraries", [])

        # Load the labels from the JSON file
        with open(labels_file, 'r') as f:
            self.raw_labels = json.load(f)
            self.labels = { key: label for key, label in self.raw_labels.items() if "reviewed" in label and label["reviewed"] }
        self.keys = list(self.labels.keys())
        self.print_label_stats()

    def print_label_stats(self):
        total_labels = len(self.labels)
        counts = {
            "crowCount": {},
            "crowAge": {},
            "quality": {},
            "alert": 0,
            "begging": 0,
            "softSong": 0,
            "rattle": 0,
            "mob": 0
        }

        for key, label in self.labels.items():
            cc = label.get("crowCount", 1)
            counts["crowCount"][cc] = counts["crowCount"].get(cc, 0) + 1

            ca = label.get("crowAge", 1)
            counts["crowAge"][ca] = counts["crowAge"].get(ca, 0) + 1

            q = label.get("quality", 2)
            if q == 3:
                q = 2  # Force value 3 (HQ) to be a 2 (for training)
            counts["quality"][q] = counts["quality"].get(q, 0) + 1

            counts["alert"] += 1 if label.get("alert", False) else 0
            counts["begging"] += 1 if label.get("begging", False) else 0
            counts["softSong"] += 1 if label.get("softSong", False) else 0
            counts["rattle"] += 1 if label.get("rattle", False) else 0
            counts["mob"] += 1 if label.get("mob", False) else 0

        print("\n=== Label Statistics ===")
        print(f"Total labels loaded: {total_labels}\n")

        print("crowCount distribution:")
        for cls in sorted(counts["crowCount"]):
            print(f"  Class {cls}: {counts['crowCount'][cls]}")

        print("\ncrowAge distribution:")
        for cls in sorted(counts["crowAge"]):
            print(f"  Class {cls}: {counts['crowAge'][cls]}")

        print("\nquality distribution:")
        for cls in sorted(counts["quality"]):
            print(f"  Class {cls}: {counts['quality'][cls]}")

        print(f"\nalert count: {counts['alert']}")
        print(f"begging count: {counts['begging']}")
        print(f"softSong count: {counts['softSong']}")
        print(f"rattle count: {counts['rattle']}")
        print(f"mob count: {counts['mob']}\n")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        file_id, start, end = key.split("-")

        # Load the per-file embedding (searching all libraries)
        emb_path = locate_embedding(file_id)
        if emb_path is None:
            raise FileNotFoundError(f"Embedding for {file_id} not found")
        embedding = np.load(emb_path)
        embedding = embedding[int(start):int(end)]

        # Convert embedding to a torch tensor.
        embedding_tensor = torch.from_numpy(embedding).float()

        # Get label info.
        label = self.labels[key]
        crowCount = label.get('crowCount', 1)
        crowAge = label.get('crowAge', 1)
        alert = 1 if label.get('alert', False) else 0
        begging = 1 if label.get('begging', False) else 0
        softSong = 1 if label.get('softSong', False) else 0
        rattle = 1 if label.get('rattle', False) else 0
        mob = 1 if label.get('mob', False) else 0
        quality = label.get('quality', 2)
        if quality == 3:
            quality = 2 # Force value 3 (HQ) to be a 2 (for training)

        return embedding_tensor, {
            "crowCount": crowCount,
            "crowAge": crowAge,
            "alert": alert,
            "begging": begging,
            "softSong": softSong,
            "rattle": rattle,
            "mob": mob,
            "quality": quality
        }
