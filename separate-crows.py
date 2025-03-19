import torch
import soundfile as sf
import torchaudio
from biodenoising.denoiser.dsp import convert_audio
from asteroid.models import DPRNNTasNet

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 22, *46, 91
# 43, *46, 49 (2 rounds of testing finds 46 the nicest to listen to)
model_checkpoint = "separator/models/best_model_epoch=46.ckpt"

# 16 is pretty good
# 20-21 is better
# 22 is better
# 26, 27, 28 is good, not sure if better than 22
# 32 seems worse
# 42 seems better
# 43 better
# 45 similar to 43
# 46 better
# 58, 64, 79 good
# 87 not sure if it's better
# 91 ???

input_audio = "samples/non-overlapping-multiple-crows.mp3"
# input_audio = "samples/overlapping-crows-1.wav"
# input_audio = "samples/overlapping-crows-2.wav"
# input_audio = "samples/overlapping-crows-3.wav"


# Manually define the model architecture (match it exactly to your training setup)
FIXED_N_SRC = 2  # Adjust this to match how many sources you separated in training
model = DPRNNTasNet(n_src=FIXED_N_SRC).to(device)  # Move model to device

# Load the PyTorch Lightning checkpoint
checkpoint = torch.load(model_checkpoint, map_location="cpu")

# Extract the actual model state_dict (Lightning saves it under "model")
if "state_dict" in checkpoint:
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
else:
    raise ValueError(f"Invalid checkpoint format! Missing 'state_dict'. Available keys: {checkpoint.keys()}")

# Load weights into the model
model.load_state_dict(state_dict)
model.eval()

# Load the MP3 file
wav, sr = torchaudio.load(input_audio)

# Convert audio to the model's expected sample rate and channel configuration
wav = convert_audio(wav, sr, model.sample_rate, model.in_channels).to(device)

# Convert shape to (batch, channels, time)
mixture = wav.unsqueeze(0)  # (1, channels, time)

# Convert to torch tensor and run inference
with torch.no_grad():
    separated_sources = model.separate(mixture)

# Convert back to NumPy and save
separated_sources = separated_sources.squeeze(0).cpu().numpy()

for i, source in enumerate(separated_sources):
    output_filename = f"separated_source_{i + 1}.wav"
    sf.write(output_filename, source, model.sample_rate)  # Ensure correct sample rate
    print(f"✅ Saved separated source: {output_filename}")
