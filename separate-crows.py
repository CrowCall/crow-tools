import torch
import soundfile as sf
import torchaudio
from biodenoising.denoiser.dsp import convert_audio
from asteroid.models import DPRNNTasNet

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_checkpoint = "separator/models/best_model-v17.ckpt"
input_audio = "samples/synthec-crows.wav"
#input_audio = "samples/multiple-crow-sounds.mp3"
#input_audio = "labeler-vue/public/mixes/merged/mix_0.wav"

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
