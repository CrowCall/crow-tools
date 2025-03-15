import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

embeddings_dir = "/home/jonathan/apps/earthspecies/ispa/crows/embeddings/"
labels_file = "/home/jonathan/apps/earthspecies/crow-sounds/labeler-vue/public/labels.json"

class CrowDataset(Dataset):
    def __init__(self):
        # Load the labels from the JSON file.
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.keys = list(self.labels.keys())

        # Define mappings for categorical labels.
        self.crowCount_map = {"single": 0, "multiple": 1, "": 0}
        self.crowAge_map = {"adult": 0, "juvenile": 1, "": 0}

        # Define the number of embeddings per 3-second segment.
        # (Assuming about 50 embeddings per second, so 3*50 = 150)
        self.chunk_size = 149

        # Build a list of chunks: (key, start_index, end_index)
        self.chunks = []
        for key in self.keys:
            cached_path = os.path.join(embeddings_dir, f"{key}.npy")
            if os.path.exists(cached_path):
                emb = np.load(cached_path)
                n_embeddings = emb.shape[0]
                num_chunks = n_embeddings // self.chunk_size  # discard leftovers
                for i in range(num_chunks):
                    start = i * self.chunk_size
                    end = start + self.chunk_size
                    self.chunks.append((key, start, end))
            else:
                # If an embedding file is missing, you might choose to skip it
                pass

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        key, start, end = self.chunks[idx]
        cached_path = os.path.join(embeddings_dir, f"{key}.npy")
        if os.path.exists(cached_path):
            emb = np.load(cached_path)
            # Take the defined 3-second chunk and compute the mean.
            chunk = emb[start:end]
            embedding = np.mean(chunk, axis=0)
        else:
            embedding = np.zeros(768)  # Fallback if file not found.
        # Convert embedding to a torch tensor.
        embedding_tensor = torch.from_numpy(embedding).float()

        # Get label info for this key.
        label = self.labels[key]
        crowCount = self.crowCount_map[label['crowCount']]
        crowAge = self.crowAge_map[label['crowAge']]
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
