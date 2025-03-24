import os
import json
import numpy as np
from classifier.classify import predict_embedding

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# For audio playback using librosa
import librosa
import sounddevice as sd

PATH = os.path.dirname(__file__)
N_CLUSTERS = 25  # number of clusters to use
embeddings_dir = os.path.join(PATH, "..", ".cache", "embeddings-denoised")


def load_embeddings():
    """
    Load all embeddings from the embeddings folder along with their segment keys, associated audio paths,
    and the segment start/end times.

    Returns:
        keys: list of segment keys.
        embeddings: numpy array of embeddings (n_samples, 768)
        audio_paths: dictionary mapping segment key to its audio file path.
        segment_info: dictionary mapping segment key to a tuple (start_time, end_time)
    """
    keys = []
    embeddings_list = []
    audio_paths = {}
    segment_info = {}

    # Load segments from segments.json.
    segments_path = os.path.join(PATH, "..", ".cache", "segments.json")
    with open(segments_path, encoding='utf-8', mode='r') as f:
        segments_dict = json.load(f)

    for file_id, segments in segments_dict.items():
        # Construct audio file path (assumes MP3 files in the public library)
        audio_path = os.path.join(PATH, "..", ".cache", "library", f"{file_id}.mp3")
        for segment in segments:
            start_time = segment.get('start_time')
            end_time = segment.get('end_time')
            # Create a unique key for the segment.
            segment_key = f"{file_id}-{int(start_time)}-{int(end_time)}"
            # Check if embedding exists:
            embedding_path = os.path.join(embeddings_dir, f"{segment_key}.npy")
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                if embedding.shape == (768,):  # ensure it is a 1D array with 768 dims
                    keys.append(segment_key)
                    embeddings_list.append(embedding)
                    audio_paths[segment_key] = audio_path
                    segment_info[segment_key] = (start_time, end_time)
                else:
                    print(f"Embedding for {segment_key} has unexpected shape {embedding.shape}. Skipping.")
            else:
                print(f"Cached embedding for segment {segment_key} not found, skipping.")

    embeddings = np.array(embeddings_list)
    return keys, embeddings, audio_paths, segment_info


def play_audio_preview(audio_file, start_time, duration):
    """
    Load and play the audio segment from the given file using librosa.

    Parameters:
        audio_file: Path to the audio file.
        start_time: The start time (in seconds) of the segment.
        duration: Duration (in seconds) to play.
    """
    try:
        # Load the specified segment using librosa.
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)
        sd.play(y, sr)
        sd.wait()
    except Exception as e:
        print(f"Failed to play audio {audio_file}: {e}")


def start_clustering(preview=False):
    # Load all embeddings along with their segment keys, audio paths, and segment info.
    keys, embeddings, audio_paths, segment_info = load_embeddings()
    if embeddings.shape[0] == 0:
        print("No embeddings found. Exiting.")
        return

    # Perform KMeans clustering directly on the 768-dimensional embeddings.
    print(f"Clustering into {N_CLUSTERS} clusters using full 768 dimensions...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Create a dictionary to collect indices for each cluster.
    cluster_indices = {i: [] for i in range(N_CLUSTERS)}
    for idx, cluster_label in enumerate(clusters):
        cluster_indices[cluster_label].append(idx)

    # Calculate distances to centroids for preview selection.
    distances = cdist(embeddings, centroids, metric='euclidean')

    # Dictionary to store auto-generated labels with cluster attribute.
    cluster_labels = {}

    # Process each cluster: select 25 examples (if available)
    for cluster_num in range(N_CLUSTERS):
        indices = cluster_indices[cluster_num]
        if not indices:
            print(f"No examples found for cluster {cluster_num + 1}")
            continue

        # Sort indices for this cluster by distance to cluster centroid.
        sorted_indices = sorted(indices, key=lambda i: distances[i, cluster_num])
        # Select top 25 indices for labeling.
        selected_indices = sorted_indices[:25]

        print(f"Processing cluster {cluster_num + 1} with {len(selected_indices)} examples...")
        for idx in selected_indices:
            segment_key = keys[idx]
            # Load the corresponding embedding (reloading from file)
            embedding_path = os.path.join(embeddings_dir, f"{segment_key}.npy")
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                predicted_label = predict_embedding(embedding)
                # Append cluster attribute (cluster numbers are 1-indexed)
                predicted_label["cluster"] = cluster_num + 1
                cluster_labels[segment_key] = predicted_label
            else:
                print(f"Embedding file missing for {segment_key}, skipping labeling.")

        # If preview mode is enabled, play the exact segment for the top 3 examples in this cluster.
        if preview:
            top3 = sorted_indices[:3]
            print(f"Previewing top 3 segments for cluster {cluster_num + 1}...")
            for idx in top3:
                segment_key = keys[idx]
                audio_file = audio_paths.get(segment_key)
                if audio_file and os.path.exists(audio_file):
                    if segment_key in segment_info:
                        start_time, end_time = segment_info[segment_key]
                        seg_duration = end_time - start_time
                        print(f"Playing segment {segment_key} from {audio_file} (start: {start_time}, duration: {seg_duration}s)")
                        play_audio_preview(audio_file, start_time, seg_duration)
                    else:
                        print(f"Segment info for {segment_key} not found.")
                else:
                    print(f"Audio file for segment {segment_key} not found.")

    # Save the cluster labels to ".cache/cluster_labels.json"
    out_path = os.path.join(PATH, "..", ".cache", "cluster_labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cluster_labels, f, indent=4)
    print(f"Saved cluster labels to {out_path}")


if __name__ == "__main__":
    # Set preview to True to enable playback of the exact segments per cluster.
    start_clustering(preview=False)
