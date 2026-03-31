#!/usr/bin/env python3
import argparse
import os
import sys

import soundfile as sf
import torch
import torchaudio
from asteroid.models import DPRNNTasNet
from biodenoising.denoiser.dsp import convert_audio


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "best_model.ckpt")
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "samples", "overlapping-crows-1.wav")
FIXED_N_SRC = 2


def load_model(checkpoint_path, device):
    model = DPRNNTasNet(n_src=FIXED_N_SRC).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format in {checkpoint_path!r}; expected a Lightning checkpoint with 'state_dict'."
        )

    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def separate_audio(input_audio, checkpoint_path=DEFAULT_CHECKPOINT, output_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)

    wav, sr = torchaudio.load(input_audio)
    wav = convert_audio(wav, sr, model.sample_rate, model.in_channels).to(device)
    mixture = wav.unsqueeze(0)

    with torch.no_grad():
        separated_sources = model.separate(mixture)

    separated_sources = separated_sources.squeeze(0).cpu().numpy()

    output_dir = os.path.abspath(output_dir or os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_audio))[0]
    written_files = []
    for i, source in enumerate(separated_sources, start=1):
        output_filename = os.path.join(output_dir, f"{stem}.source-{i}.wav")
        sf.write(output_filename, source, model.sample_rate)
        written_files.append(output_filename)
        print(f"Saved separated source: {output_filename}")

    return written_files


def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate overlapping crow calls from an input audio file.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to the input audio file. Defaults to the bundled sample clip.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the separator model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for separated output files. Defaults to the current working directory.",
    )
    args = parser.parse_args(argv)

    separate_audio(
        args.audio_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
