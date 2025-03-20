import os
import json
import numpy as np
from classifier.classify import predict_embedding

PATH = os.path.dirname(__file__)


def main():
    # Path to segments (we use segments.json to iterate over all segments).
    segments_path = os.path.join("labeler-vue", "public", "segments.json")

    # Load segments.
    with open(segments_path, encoding='utf-8', mode='r') as f:
        segments_dict = json.load(f)

    # Dictionary to store auto-generated labels.
    auto_labels = {}

    # Iterate over segments.
    for file_id, segments in segments_dict.items():
        for segment in segments:
            # Construct segment key from file_id and (integer) start/end times.
            start_time = segment.get('start_time')
            end_time = segment.get('end_time')
            segment_key = f"{file_id}-{int(start_time)}-{int(end_time)}"

            # Build the audio file path (assumes MP3 files in the public library).
            audio_path = os.path.join(PATH, "labeler-vue", "public", "library", f"{file_id}.mp3")

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found, skipping segment {segment_key}.")
                continue

            # Cached embedding file path.
            embedding_path = os.path.join("embeddings", f"{segment_key}.npy")
            if os.path.exists(embedding_path):
                # Load the cached embedding.
                embedding = np.load(embedding_path)
                # Pass the embedding to our classifier to get predicted label dict.
                predicted_label = predict_embedding(embedding)
                auto_labels[segment_key] = predicted_label
                print(f"Auto-labeled segment {segment_key}.")
            else:
                print(f"Cached embedding for segment {segment_key} not found, skipping.")

    # Save the auto labels to "auto_labels.json".
    with open(os.path.join(PATH, "labeler-vue", "public", "auto_labels.json"), "w", encoding="utf-8") as f:
        json.dump(auto_labels, f, indent=4)
    print("Saved auto labels to auto_labels.json")


if __name__ == "__main__":
    main()
