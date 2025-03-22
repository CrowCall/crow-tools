import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

PATH = os.path.dirname(__file__)

embeddings_dir = os.path.join(PATH, "..", ".cache", "embeddings")
labels_file = os.path.join(PATH, ".. ", ".cache", "labels.json")

class CrowDataset(Dataset):
    def __init__(self):
        # Load the labels from the JSON file.
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.keys = list(self.labels.keys())

        # Define mappings for categorical labels.
        self.crowCount_map = {"single": 0, "multiple": 1}
        self.crowAge_map = {"adult": 0, "juvenile": 1}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # Load embedding
        cached_path = os.path.join(embeddings_dir, f"{key}.npy")
        if os.path.exists(cached_path):
            embedding = np.load(cached_path)
        else:
            embedding = np.zeros(768)  # Fallback if file not found.
        # Convert embedding to a torch tensor.
        embedding_tensor = torch.from_numpy(embedding).float()

        # Get label info for this key.
        label = self.labels[key]
        crowCount = self.crowCount_map[label['crowCount'] or 'single']
        crowAge = self.crowAge_map[label['crowAge'] or 'adult']
        begging = 1 if label['begging'] else 0
        softSong = 1 if label['softSong'] else 0
        rattle = 1 if label['rattle'] else 0
        badQuality = 1 if label['badQuality'] else 0
        human = 1 if label['human'] else 0

        return embedding_tensor, {
            "crowCount": crowCount,
            "crowAge": crowAge,
            "begging": begging,
            "softSong": softSong,
            "rattle": rattle,
            "badQuality": badQuality,
            "human": human
        }
