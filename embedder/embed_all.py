import os
import librosa
import numpy as np
from embedder.embed import generate_embeddings
from embedder.ispa import utils

PATH = os.path.dirname(__file__)

def start_embeddings(denoised=False):
    DENOISED_MODE = ""
    EXT = ".mp3"
    if denoised:
        DENOISED_MODE = "-denoised"
        EXT = ".wav"

    # Directory with the denoised .wav files
    library_path = os.path.join(PATH, "..", ".cache", f"library{DENOISED_MODE}")
    file_paths = sorted(os.listdir(library_path))

    # Where to save the averaged embeddings
    embeddings_path = os.path.join(PATH, "..", ".cache", f"embeddings{DENOISED_MODE}")
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)

    # Where to save per-second volume data
    volumes_path = os.path.join(PATH, "..", ".cache", f"embeddings{DENOISED_MODE}-volumes")
    if not os.path.exists(volumes_path):
        os.makedirs(volumes_path)

    sample_rate = 8000
    chunk_size = 25  # Number of frames to average into one vector

    for file_name in file_paths:
        if not file_name.endswith(EXT):
            continue

        file_id = os.path.splitext(file_name)[0]
        audio_path = os.path.join(library_path, file_name)

        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found, skipping {file_id}.")
            continue

        # Embedding output
        embedding_out_path = os.path.join(embeddings_path, f"{file_id}.npy")
        # Volume output
        volume_out_path = os.path.join(volumes_path, f"{file_id}.npy")

        # Skip if both already exist
        if os.path.exists(embedding_out_path) and os.path.exists(volume_out_path):
            print(f"Skipping {file_id}; embedding & volume both exist.")
            continue

        print(f"Loading {audio_path} at {sample_rate} Hz.")
        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        if sr != sample_rate:
            print(f"Warning: Sample rate mismatch for {file_id}. Using SR={sr}.")

        ############################################################
        # 1) Generate and save embeddings (if not already existing)
        ############################################################
        if not os.path.exists(embedding_out_path):
            full_embedding = generate_embeddings(waveform)  # shape: (N_frames, 768)
            full_embedding = np.array(full_embedding, dtype=np.float32)
            num_frames = full_embedding.shape[0]
            print(f"{file_id}: got {num_frames} frames of embeddings.")

            # Chunk the frames in increments of 'chunk_size' and average
            num_chunks = int(np.ceil(num_frames / chunk_size))
            embedding_means = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_frames)
                chunk = full_embedding[start_idx:end_idx]
                mean_emb = np.mean(chunk, axis=0)
                embedding_means.append(mean_emb)
            final_embedding = np.stack(embedding_means, axis=0)
            np.save(embedding_out_path, final_embedding)
            print(f"Saved embedding for {file_id} -> {embedding_out_path} shape={final_embedding.shape}")

        ############################################################
        # 2) Compute and save volume data (if not existing)
        ############################################################
        if not os.path.exists(volume_out_path):
            # Ensure waveform is 1D. librosa.load returns a 1D array if mono=True,
            # but in case it ends up 2D, squeeze the first axis.
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform, axis=0)
            total_samples = waveform.shape[0]

            # Compute 1 volume metric per second (mean absolute amplitude)
            total_seconds = int(np.ceil(total_samples / sr))
            volumes = []
            for sec in range(total_seconds):
                start_samp = sec * sr
                end_samp = min((sec + 1) * sr, total_samples)
                segment = waveform[start_samp:end_samp]
                if len(segment) == 0:
                    volumes.append(0.0)
                else:
                    mean_amp = np.mean(np.abs(segment))
                    volumes.append(mean_amp)

            volumes = np.array(volumes, dtype=np.float32)
            np.save(volume_out_path, volumes)
            print(f"Saved volume data for {file_id} -> {volume_out_path} shape={volumes.shape}")

if __name__ == "__main__":
    start_embeddings()
