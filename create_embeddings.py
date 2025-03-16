import os
import json
import numpy as np
from aves.embed import generate_embeddings
from aves.ispa import utils

def main():
    # Paths to segments and labels.
    segments_path = "labeler-vue/public/segments.json"

    with open(segments_path, encoding='utf-8', mode='r') as f:
        segments_dict = json.load(f)

    # Define parameters.
    sample_rate = 8000

    # Iterate over segments.
    for file_id, segments in segments_dict.items():
        for segment in segments:
            # Construct a segment key using file_id and the (integer) start/end times.
            start_time = segment.get('start_time')
            end_time = segment.get('end_time')
            segment_key = f"{file_id}-{int(start_time)}-{int(end_time)}"

            # Build the audio file path.
            # Adjust the paths as needed. Here we assume non-denoised files in MP3 format.
            audio_path = os.path.join( "/home/jonathan/apps/earthspecies/crow-sounds/labeler-vue/public/library", f"{file_id}.mp3")

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found, skipping segment {segment_key}.")
                continue

            # Generate the embedding.
            embedding_path = os.path.join("embeddings", f"{segment_key}.npy")
            if not os.path.exists(embedding_path):
                # Load the waveform for this 3-second segment.
                print(f"Load wav file {audio_path}.")
                waveform, sr = utils.load_waveform(audio_path, tgt_sr=sample_rate, start_sec=start_time, end_sec=end_time)
                if sr != sample_rate:
                    print(f"Warning: Sample rate mismatch for {file_id}. Using loaded sample rate {sr}.")

                embedding = generate_embeddings(waveform)
                np.save(embedding_path, embedding)
                print(f"Saved embedding for {segment_key} to {embedding_path}")
            else:
                print(f"Skipping embedding for {segment_key}.")

if __name__ == "__main__":
    main()
