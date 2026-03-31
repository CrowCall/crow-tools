import os
import json
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import sys

PATH = os.path.dirname(__file__)
ROOT = os.path.dirname(PATH)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowtools.datasets import get_cache_base, get_dataset_libraries, get_libraries_base, resolve_dataset_embedding_path

TRAIN_SPLIT_FRACTION = 0.88
DATASET_SPLIT_SEED = 18202


def locate_embedding(file_id, dataset_name, libraries, denoised=True, cache_base=None):
    """Return path to the embedding for ``file_id`` in an allowed library."""
    dataset_path = resolve_dataset_embedding_path(
        dataset_name,
        file_id,
        denoised=denoised,
        cache_base=cache_base,
    )
    if dataset_path is not None:
        return dataset_path

    suffix = "embeddings-denoised" if denoised else "embeddings"
    libraries_base = get_libraries_base(cache_base)
    for lib in libraries:
        emb_dir = os.path.join(libraries_base, lib, suffix)
        path = os.path.join(emb_dir, f"{file_id}.npy")
        if os.path.exists(path):
            return path
    return None


def split_train_val_dataset(dataset, train_fraction=TRAIN_SPLIT_FRACTION, seed=DATASET_SPLIT_SEED):
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

class CrowDataset(Dataset):
    def __init__(self, dataset_name="all-public", cache_base=None):
        self.dataset_name = dataset_name
        self.cache_base = get_cache_base(cache_base)
        dataset_dir = os.path.join(self.cache_base, "datasets", dataset_name)
        labels_file = os.path.join(dataset_dir, "labels.json")
        config_file = os.path.join(dataset_dir, "config.json")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels not found for dataset {dataset_name}")

        self.included_libraries = get_dataset_libraries(dataset_name, self.cache_base)
        if os.path.exists(config_file) and not self.included_libraries:
            with open(config_file, 'r') as cf:
                cfg = json.load(cf)
                self.included_libraries = cfg.get("included_libraries", [])

        # Load the labels from the JSON file
        with open(labels_file, 'r') as f:
            self.raw_labels = json.load(f)
            self.labels = { key: label for key, label in self.raw_labels.items() if "reviewed" in label and label["reviewed"] }
        self.keys = sorted(self.labels.keys())
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
        emb_path = locate_embedding(
            file_id,
            dataset_name=self.dataset_name,
            libraries=self.included_libraries,
            cache_base=self.cache_base,
        )
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
