import os
import torch
import torchaudio
import librosa
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio
from tqdm import tqdm

from crowtools.datasets import get_library_dir, get_public_libraries

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = os.path.dirname(__file__)

# Load the pre-trained model
model = pretrained.biodenoising16k_dns48().to(device)

def start_denoising(libraries=None, selected_ids_by_library=None, cache_base=None):
    """Denoise all audio files in each library."""
    libraries = list(libraries) if libraries is not None else get_public_libraries(cache_base)
    for lib in libraries:
        library_base = get_library_dir(lib, cache_base)
        library_dir = os.path.join(library_base, "audio")
        output_dir = os.path.join(library_base, "audio-denoised")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(library_dir):
            continue

        files = sorted(os.listdir(library_dir))
        selected_ids = None if selected_ids_by_library is None else selected_ids_by_library.get(lib)

        for filename in files:
            if filename.lower().endswith('.mp3'):
                base_name = os.path.splitext(filename)[0]
                if selected_ids is not None and base_name not in selected_ids:
                    continue
                output_file = os.path.join(output_dir, f"{base_name}.wav")

                # Skip if the output file already exists
                if os.path.exists(output_file):
                    print(f"Skipping {filename} (denoised file exists)")
                    continue

                input_file = os.path.join(library_dir, filename)
                print(f"Processing: {input_file}")

                # Load the MP3 file using librosa at 16kHz mono.
                # librosa.load returns a NumPy array of shape (samples,)
                audio_np, sr = librosa.load(input_file, sr=16000, mono=True)
                if sr != 16000:
                    print(f"Warning: Sample rate mismatch for {filename}. Using SR={sr}.")
                wav = torch.from_numpy(audio_np).unsqueeze(0)

                # Convert audio to the model's expected sample rate and channel configuration
                wav = convert_audio(wav, sr, model.sample_rate, model.chin).to(device)

                total_samples = wav.size(1)
                chunk_size = model.sample_rate * 60
                if total_samples > chunk_size:
                    print(f"Splitting {filename} into chunks...")
                    denoised_chunks = []
                    for start in tqdm(range(0, total_samples, chunk_size)):
                        end = min(start + chunk_size, total_samples)
                        chunk = wav[:, start:end]
                        with torch.no_grad():
                            chunk_denoised = model(chunk.unsqueeze(0))[0]
                        denoised_chunks.append(chunk_denoised)
                    denoised = torch.cat(denoised_chunks, dim=1)
                else:
                    with torch.no_grad():
                        denoised = model(wav.unsqueeze(0))[0]

                torchaudio.save(output_file, denoised.cpu(), model.sample_rate)
                print(f"Saved denoised audio to: {output_file}")

if __name__ == "__main__":
    start_denoising()
