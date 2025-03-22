import os
import torch
import torchaudio
import numpy as np
import json

FIXED_N_SRC = 2


class CrowMixDataset:
    def __init__(self, json_path, merged_dir, separate_dir, sr=8000, transform=None):
        # Load the full mixes dataset.
        with open(json_path, "r", encoding="utf-8") as f:
            self.mixes = json.load(f)

        print(f"Found {len(self.mixes)} mixes.")

        self.merged_dir = merged_dir
        self.separate_dir = separate_dir
        self.sr = sr
        self.transform = transform

    def __len__(self):
        return len(self.mixes)

    def pad_sources(self, sources, n_src, expected_length):
        # sources: list of tensors of shape (channels, time)
        # expected_length: target time dimension length
        while len(sources) < n_src:
            # Create a silent source with the same number of channels (assume mono here) and expected length
            silence = torch.zeros(1, expected_length)
            sources.append(silence)
        # Stack along a new dimension; shape becomes (n_src, channels, time)
        stacked = torch.stack(sources, dim=0)
        # If sources are mono (channels == 1), squeeze that dimension to get shape (n_src, time)
        if stacked.shape[1] == 1:
            stacked = stacked.squeeze(1)
        return stacked

    def __getitem__(self, idx):
        mix_info = self.mixes[idx]
        mix_path = os.path.join(self.merged_dir, mix_info["mix"])
        mixture, sr_m = torchaudio.load(mix_path)
        if sr_m != self.sr:
            mixture = torchaudio.functional.resample(mixture, sr_m, self.sr)

        source_keys = sorted(mix_info["layers"].keys())[1:]
        sources = []
        for key in source_keys:
            src_filename = mix_info["layers"][key]
            src_path = os.path.join(self.separate_dir, src_filename)
            source, sr_s = torchaudio.load(src_path)
            if sr_s != self.sr:
                source = torchaudio.functional.resample(source, sr_s, self.sr)
            sources.append(source)

        # Determine expected length from the mixture
        expected_length = mixture.shape[1]

        # Pad if necessary
        sources = self.pad_sources(sources, FIXED_N_SRC, expected_length)

        return mixture, sources

def collate_fn(batch):
    mixtures, sources_list = zip(*batch)  # Unpack batch tuples

    # Convert to numpy arrays before torch conversion (avoids PyTorch warning)
    mixtures = [mixture.numpy() if isinstance(mixture, torch.Tensor) else mixture for mixture in mixtures]
    sources_list = [sources.numpy() if isinstance(sources, torch.Tensor) else sources for sources in sources_list]

    # Determine max time length among mixtures
    max_len = max(mixture.shape[1] for mixture in mixtures)

    # Preallocate tensors
    batch_size = len(batch)
    padded_mixtures = np.zeros((batch_size, mixtures[0].shape[0], max_len), dtype=np.float32)  # [batch, channels, time]
    padded_sources = np.zeros((batch_size, FIXED_N_SRC, max_len), dtype=np.float32)  # [batch, FIXED_N_SRC, time]

    for i, (mixture, sources) in enumerate(zip(mixtures, sources_list)):
        # Copy mixture data (instead of appending, preallocated)
        padded_mixtures[i, :, :mixture.shape[1]] = mixture

        # Squeeze channel dimension if sources are mono
        if sources.ndim == 3 and sources.shape[1] == 1:
            sources = sources.squeeze(1)  # Shape [n_src, time]

        # Pad sources in time
        sources_padded = np.zeros((FIXED_N_SRC, max_len), dtype=np.float32)
        sources_padded[:sources.shape[0], :sources.shape[1]] = sources[:FIXED_N_SRC]  # Truncate or pad

        padded_sources[i] = sources_padded  # Assign to preallocated array

    # Convert final arrays to tensors
    padded_mixtures = torch.tensor(padded_mixtures)
    padded_sources = torch.tensor(padded_sources)

    return padded_mixtures, padded_sources
