import os
import torch
import torchaudio
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = os.path.dirname(__file__)

# Load the pre-trained model
model = pretrained.biodenoising16k_dns48().to(device)

def start_denoising():
    # Define directories
    library_dir = os.path.join(PATH, "..", ".cache", "library")
    output_dir = os.path.join(PATH, "..", ".cache", "library-denoised")
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

            # Convert audio to the model's expected sample rate and channel configuration
            wav = convert_audio(wav, sr, model.sample_rate, model.chin).to(device)

            # Determine if we need to split the file (if longer than 1 minute)
            total_samples = wav.size(1)
            chunk_size = model.sample_rate * 60  # samples in 60 seconds
            if total_samples > chunk_size:
                print(f"Splitting {filename} into chunks...")
                denoised_chunks = []
                # Process each chunk individually
                for start in range(0, total_samples, chunk_size):
                    end = min(start + chunk_size, total_samples)
                    chunk = wav[:, start:end]
                    with torch.no_grad():
                        chunk_denoised = model(chunk[None])[0]
                    denoised_chunks.append(chunk_denoised)
                # Reassemble the denoised chunks along the time dimension
                denoised = torch.cat(denoised_chunks, dim=1)
            else:
                # Process the whole file if it is 1 minute or shorter
                with torch.no_grad():
                    denoised = model(wav[None])[0]

            # Save the denoised audio as a WAV file
            torchaudio.save(output_file, denoised.cpu(), model.sample_rate)
            print(f"Saved denoised audio to: {output_file}")

if __name__ == "__main__":
    start_denoising()