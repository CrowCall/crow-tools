import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

FIXED_N_SRC = 5


class CrowMixDataset(Dataset):
    def __init__(self, json_path, merged_dir, separate_dir, sr=16000, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.mixes = json.load(f)
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
        return torch.stack(sources, dim=0)

    # In your __getitem__:
    def __getitem__(self, idx):
        mix_info = self.mixes[idx]
        mix_path = os.path.join(self.merged_dir, mix_info["mix"])
        mixture, sr_m = torchaudio.load(mix_path)
        if sr_m != self.sr:
            mixture = torchaudio.functional.resample(mixture, sr_m, self.sr)

        source_keys = sorted(mix_info["layers"].keys())
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
        # Set n_src to the maximum number of sources (for example, 3)
        n_src = 3
        # Pad if necessary
        sources = self.pad_sources(sources, n_src, expected_length)

        return mixture, sources


def collate_fn(batch):
    mixtures, sources_list = zip(*batch)

    # Determine max time length among mixtures
    max_len = max(mixture.shape[1] for mixture in mixtures)

    padded_mixtures = []
    padded_sources = []

    for mixture, sources in zip(mixtures, sources_list):
        # Pad mixture (assumed shape [channels, time])
        pad_time = max_len - mixture.shape[1]
        if pad_time > 0:
            mixture = F.pad(mixture, (0, pad_time))
        padded_mixtures.append(mixture)

        # Assume sources is [n_src, channels, time]. If channels==1, squeeze it.
        if sources.shape[1] == 1:
            sources = sources.squeeze(1)  # now shape [n_src, time]
        # Pad sources in time dimension
        pad_time = max_len - sources.shape[1]
        if pad_time > 0:
            sources = F.pad(sources, (0, pad_time))
        # Now pad or trim along the n_src dimension
        current_n_src = sources.shape[0]
        if current_n_src < FIXED_N_SRC:
            pad_tensor = torch.zeros((FIXED_N_SRC - current_n_src, sources.shape[1]), dtype=sources.dtype)
            sources = torch.cat([sources, pad_tensor], dim=0)
        elif current_n_src > FIXED_N_SRC:
            sources = sources[:FIXED_N_SRC]
        padded_sources.append(sources)

    # Stack mixtures: if mixtures were mono, shape becomes [batch, time]
    padded_mixtures = torch.stack(padded_mixtures)
    # Stack sources: shape becomes [batch, FIXED_N_SRC, time]
    padded_sources = torch.stack(padded_sources)
    return padded_mixtures, padded_sources
