import argparse
import os
import sys
import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embedder.embed import generate_embeddings
from embedder.ispa import utils
from crowtools.datasets import get_dataset_libraries, get_library_dir, get_public_libraries, get_selected_files

PATH = os.path.dirname(__file__)

def start_embeddings(denoised=False, libraries=None, selected_ids_by_library=None, cache_base=None):
    DENOISED_MODE = ""
    EXT = ".mp3"
    if denoised:
        DENOISED_MODE = "-denoised"
        EXT = ".wav"

    libraries = list(libraries) if libraries is not None else get_public_libraries(cache_base)

    sample_rate = 8000
    chunk_size = 25  # Number of frames to average into one vector

    for lib in libraries:
        library_base = get_library_dir(lib, cache_base)
        library_path = os.path.join(library_base, f"audio{DENOISED_MODE}")
        if not os.path.exists(library_path):
            continue
        file_paths = sorted(os.listdir(library_path))
        selected_ids = None if selected_ids_by_library is None else selected_ids_by_library.get(lib)

        # Where to save the averaged embeddings
        embeddings_path = os.path.join(library_base, f"embeddings{DENOISED_MODE}")
        os.makedirs(embeddings_path, exist_ok=True)

        # Where to save per-second volume data
        volumes_path = os.path.join(library_base, f"embeddings{DENOISED_MODE}-volumes")
        os.makedirs(volumes_path, exist_ok=True)

        for file_name in file_paths:
            if not file_name.endswith(EXT):
                continue

            file_id = os.path.splitext(file_name)[0]
            if selected_ids is not None and file_id not in selected_ids:
                continue
            audio_path = os.path.join(library_path, file_name)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found, skipping {file_id}.")
                continue

            embedding_out_path = os.path.join(embeddings_path, f"{file_id}.npy")
            volume_out_path = os.path.join(volumes_path, f"{file_id}.npy")

            if os.path.exists(embedding_out_path) and os.path.exists(volume_out_path):
                print(f"Skipping {file_id}; embedding & volume both exist.")
                continue

            print(f"Loading {audio_path} at {sample_rate} Hz.")
            waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            if sr != sample_rate:
                print(f"Warning: Sample rate mismatch for {file_id}. Using SR={sr}.")

            if not os.path.exists(embedding_out_path):
                full_embedding = generate_embeddings(waveform)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate embeddings across a dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset to process. Defaults to all discovered public libraries.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    args = parser.parse_args(argv)

    if args.dataset:
        libraries = get_dataset_libraries(args.dataset, args.cache_dir)
        selected_ids_by_library = get_selected_files(args.dataset, args.cache_dir)
    else:
        libraries = None
        selected_ids_by_library = None

    start_embeddings(
        denoised=False,
        libraries=libraries,
        selected_ids_by_library=selected_ids_by_library,
        cache_base=args.cache_dir,
    )
    start_embeddings(
        denoised=True,
        libraries=libraries,
        selected_ids_by_library=selected_ids_by_library,
        cache_base=args.cache_dir,
    )


if __name__ == "__main__":
    main()
