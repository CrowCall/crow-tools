import sys
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")

def load_waveform(path, tgt_sr=16_000, min_duration=None, max_duration=None, start_sec=None, end_sec=None):
    waveform, src_sr = torchaudio.load(path)
    # downsample to 16kHz
    if src_sr != tgt_sr:
        print(f"Resample from {src_sr} to {tgt_sr}", file=sys.stderr)
        waveform = torchaudio.transforms.Resample(src_sr, tgt_sr)(waveform)

    # downmix if stereo
    if len(waveform) > 1:
        print(f"Downmix from {len(waveform)} channels to 1", file=sys.stderr)
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # extract segment if start_sec or end_sec are provided
    if start_sec is not None or end_sec is not None:
        start_sample = int(start_sec * tgt_sr) if start_sec is not None else 0
        end_sample = int(end_sec * tgt_sr) if end_sec is not None else waveform.shape[1]
        waveform = waveform[:, start_sample:end_sample]

    # pad to min_duration
    if min_duration is not None and len(waveform[0]) < min_duration * tgt_sr:
        print(f"Pad from {len(waveform[0]) / tgt_sr} to {min_duration} secs", file=sys.stderr)
        waveform = torch.nn.functional.pad(waveform, (0, int(min_duration * tgt_sr - len(waveform[0]))))

    # truncate to max_duration
    if max_duration is not None and len(waveform[0]) > max_duration * tgt_sr:
        print(f"Truncate from {len(waveform[0]) / tgt_sr} to {max_duration} secs", file=sys.stderr)
        waveform = waveform[:, :int(max_duration * tgt_sr)]

    return waveform, tgt_sr
