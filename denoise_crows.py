import os
import torch
import torchaudio
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = pretrained.biodenoising16k_dns48().to(device)

# Define directories
library_dir = 'labeler-vue/public/library'
output_dir = 'labeler-vue/public/denoised'
os.makedirs(output_dir, exist_ok=True)

# Get sorted list of files from the library directory
files = sorted(os.listdir(library_dir))

for filename in files:
    if filename.lower().endswith('.mp3'):
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}.wav")

        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {filename} (denoised file exists)")
            continue

        input_file = os.path.join(library_dir, filename)
        print(f"Processing: {input_file}")

        # Load the MP3 file
        wav, sr = torchaudio.load(input_file)
        wav_minutes = len(wav[0])/sr/60.0
        if wav_minutes > 10:
            print(f"Skipping {input_file} (too long)")
            continue

        # Convert audio to model's expected sample rate and channel configuration
        wav = convert_audio(wav, sr, model.sample_rate, model.chin).to(device)

        # Denoise the audio
        with torch.no_grad():
            denoised = model(wav[None])[0]

        # Save the denoised audio as a WAV file
        torchaudio.save(output_file, denoised.cpu(), model.sample_rate)
        print(f"Saved denoised audio to: {output_file}")
