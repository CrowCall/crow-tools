import os
import json
import random
import numpy as np
import librosa
import sounddevice as sd
from sklearn.decomposition import PCA
import faiss
from collections import defaultdict
import argparse
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowtools.datasets import (
    build_dataset_file_index,
    get_dataset_artifact_path,
    resolve_dataset_audio_path,
)

# Parameters
STARTING_CLUSTER_ID = 65
VOLUME_THRESHOLD = 0.0002
SUBSAMPLE_FACTOR = 1.0
PCA_COMPONENTS = 75
MAX_CLUSTER_SIZE = 500  # Maximum leaf size before splitting stops.
MERGE_THRESHOLD = 0.15  # Merge leaves if cosine distance < 0.15 (single pass)
PREVIEW_SEEDS = False
PREVIEW_CLUSTERS = False
ONLY_OUTPUT_SEEDS = True
PREVIEW_PER_CLUSTER = 10
NUM_REPRESENTATIVE = 10  # Number of segments per merged cluster for labeling
CURRENT_DATASET = "all-public"
CACHE_BASE = None

# Seed examples for new clusters.
SEED_EXAMPLES = [
    # Rattles
    #{"file_id": "365208991", "start": 13.0, "end": 14.0}, # Good, 9 similar
    #{"file_id": "227497211", "start": 48.0, "end": 49.0}, # Good, 7 similar
    #{"file_id": "124568031", "start": 7.0, "end": 8.0}, # Good 10+ similar, (had to lower volume to 0.0002)
    #{"file_id": "431165421", "start": 12.0, "end": 13.0},  # Okay, 3 similar
    #{"file_id": "58460", "start": 12.0, "end": 13.0},  # Good, 8 similar
    #{"file_id": "504976401", "start": 4.0, "end": 5.0},  # Great, 9 similar
    #{"file_id": "122364731", "start": 3.0, "end": 4.0},  # Okay, 3 similar
    #{"file_id": "163637", "start": 1.0, "end": 2.0}, # Okay, 3-4 similar
    #{"file_id": "496356", "start": 9.0, "end": 10.0}, # Good, 5 similar
    #{"file_id": "156527", "start": 30.0, "end": 31.0}, # Okay, 3 or 4 similar



{"file_id": "227497211", "start": 48.0, "end": 49.0},
{"file_id": "365208991", "start": 27.0, "end": 28.0},
{"file_id": "361178511", "start": 27.0, "end": 28.0},
{"file_id": "361178511", "start": 13.0, "end": 14.0},
{"file_id": "156527", "start": 6.0, "end": 7.0},
{"file_id": "92055", "start": 15.0, "end": 16.0},
{"file_id": "619206184", "start": 17.0, "end": 18.0},

    # Sub/Soft Song


{"file_id": "229159", "start": 74.0, "end": 75.0},
{"file_id": "542024451", "start": 7.0, "end": 8.0},
{"file_id": "539550101", "start": 12.0, "end": 13.0},
{"file_id": "535466271", "start": 1.0, "end": 2.0},
{"file_id": "408950861", "start": 21.0, "end": 22.0},
{"file_id": "319547721", "start": 100.0, "end": 101.0},
{"file_id": "167792", "start": 20.0, "end": 21.0},

    #{"file_id": "408950861", "start": 20.0, "end": 21.0},
    #{"file_id": "13123", "start": 34.0, "end": 35.0},
    #{"file_id": "984442", "start": 33.0, "end": 34.0},
    #{"file_id": "984442", "start": 6.0, "end": 7.0},

    # Juvenile begging
    #{"file_id": "32684421", "start": 39.0, "end": 40.0},
    #{"file_id": "32684421", "start": 22.0, "end": 23.0},
    #{"file_id": "32684421", "start": 10.0, "end": 11.0},
    #{"file_id": "32684421", "start": 3.0, "end": 4.0}
]


def play_audio_preview(audio_file, start_time, duration):
    try:
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)
        sd.play(y, sr)
        sd.wait()
    except Exception as e:
        print(f"Error playing {audio_file}: {e}")


def dataset_audio_preview_path(file_id):
    return resolve_dataset_audio_path(
        CURRENT_DATASET,
        file_id,
        denoised=True,
        cache_base=CACHE_BASE,
    )


def load_non_silent_embeddings(dataset_name, thresh, cache_base=None):
    emb_list, ids, total = [], [], 0
    embedding_index = build_dataset_file_index(
        dataset_name,
        relative_dir="embeddings-denoised",
        extensions=[".npy"],
        cache_base=cache_base,
    )
    volume_index = build_dataset_file_index(
        dataset_name,
        relative_dir="embeddings-denoised-volumes",
        extensions=[".npy"],
        cache_base=cache_base,
    )
    for file_id in sorted(embedding_index):
        emb_path = embedding_index[file_id]["paths"][".npy"]
        vol_record = volume_index.get(file_id)
        if not vol_record:
            continue
        vol_path = vol_record["paths"][".npy"]
        try:
            emb_data = np.load(emb_path)
            vol_data = np.load(vol_path)
            if vol_data.ndim > 1:
                vol_data = vol_data.squeeze(-1)
        except Exception:
            continue
        total += emb_data.shape[0]
        for i in range(emb_data.shape[0]):
            if vol_data[i] > thresh:
                emb_list.append(emb_data[i])
                ids.append((file_id, i))
    return np.array(emb_list, dtype=np.float32), ids, total


def random_subsample(embeddings, ids, factor):
    seed_file_ids = {seed["file_id"] for seed in SEED_EXAMPLES}
    seed_indices = [i for i, (file_id, _) in enumerate(ids) if file_id in seed_file_ids]
    other_indices = [i for i, (file_id, _) in enumerate(ids) if file_id not in seed_file_ids]
    if factor >= 1.0:
        selected_other = other_indices
    else:
        n_keep = int(len(other_indices) * factor)
        selected_other = sorted(random.sample(other_indices, n_keep))
    selected_indices = sorted(seed_indices + selected_other)
    return embeddings[selected_indices], [ids[i] for i in selected_indices]


def reduce_dim_pca(embeddings, n_components=64):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print("Explained variance (first 5):", pca.explained_variance_ratio_[:5])
    return reduced, pca


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embeddings / norms


def hierarchical_split(embeddings, indices, max_size=500, level=0):
    if len(indices) <= max_size:
        return {"indices": indices, "level": level, "size": len(indices)}
    k = 2
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, seed=42)
    subset = embeddings[indices]
    kmeans.train(subset)
    D, I = kmeans.index.search(subset, 1)
    assignments = I.flatten()
    clusters = {i: [] for i in range(k)}
    for j, assign in enumerate(assignments):
        clusters[assign].append(indices[j])
    result = {"level": level, "size": len(indices), "children": {}}
    for i in range(k):
        result["children"][i] = hierarchical_split(embeddings, clusters[i], max_size, level + 1)
    return result


def collect_leaves(hierarchy):
    leaves = []
    if "children" not in hierarchy:
        leaves.append(hierarchy)
    else:
        for child in hierarchy["children"].values():
            leaves.extend(collect_leaves(child))
    return leaves


def compute_leaf_center(norm_emb, indices):
    center = np.mean(norm_emb[indices], axis=0)
    center /= np.linalg.norm(center)
    return center


def merge_leaves_once(leaves, merges):
    merged = set()
    new_leaves = []
    for (i, j, dist) in merges:
        if i not in merged and j not in merged:
            union_indices = leaves[i]["indices"] + leaves[j]["indices"]
            new_leaf = {"indices": union_indices,
                        "level": min(leaves[i]["level"], leaves[j]["level"]),
                        "size": len(union_indices)}
            new_leaves.append(new_leaf)
            merged.add(i)
            merged.add(j)
    for i in range(len(leaves)):
        if i not in merged:
            new_leaves.append(leaves[i])
    return new_leaves


def process_seed_examples(norm_emb, ids, faiss_index, max_per_file=4):
    # Process each seed example individually.
    seed_clusters = {}
    default_template = {
        "crowCount": 1, "crowAge": 1,
        "alert": False, "begging": False,
        "softSong": False, "rattle": False, "mob": False,
        "quality": 2, "reviewed": False
    }
    for i, seed in enumerate(SEED_EXAMPLES):
        target_index = None
        for idx, (file_id, sec) in enumerate(ids):
            if file_id == seed["file_id"] and int(sec) == int(seed["start"]):
                target_index = idx
                break
        if target_index is None:
            print(f"Seed example {seed} not found.")
            continue
        vec = norm_emb[target_index].reshape(1, norm_emb.shape[1])
        D, I = faiss_index.search(vec, NUM_REPRESENTATIVE)
        results = []
        file_counts = defaultdict(int)
        for file_id, sec in [ids[idx] for idx in I[0]]:
            if file_counts[file_id] < max_per_file:
                results.append({"file_id": file_id, "start_time": float(sec), "end_time": float(sec + 1)})
                file_counts[file_id] += 1
        if results:
            seed_clusters[str(i)] = results
    return seed_clusters


def build_and_save_clusters(merged_leaves, seed_clusters, ids, norm_emb):
    """
    Build segments and labels from both merged clusters and seed clusters.
    Existing segments and labels are loaded (if available) and new records are
    appended only if they are not already present.
    - For merged clusters, segments are sorted in similarity order for preview,
      then the top representative segments are chosen.
    - For seed clusters, segments are added if not already present.
    - Segments in each file are sorted by start_time before saving.
    - Duplicate segment keys (file, start, end) are avoided.
    """
    from collections import defaultdict

    # Load existing segments and labels if the files exist.
    output_segments = get_dataset_artifact_path(CURRENT_DATASET, "segments.json", cache_base=CACHE_BASE)
    output_labels = get_dataset_artifact_path(CURRENT_DATASET, "labels.json", cache_base=CACHE_BASE)
    label_template_file = get_dataset_artifact_path(CURRENT_DATASET, "cluster_segments_labels.json", cache_base=CACHE_BASE)

    if os.path.exists(output_segments):
        with open(output_segments, "r") as f:
            existing_segments = json.load(f)
    else:
        existing_segments = {}

    if os.path.exists(output_labels):
        with open(output_labels, "r") as f:
            existing_labels = json.load(f)
    else:
        existing_labels = {}

    # New segments and labels to add.
    new_segments = defaultdict(list)
    new_labels = {}

    if os.path.exists(label_template_file):
        with open(label_template_file, "r") as f:
            label_templates = json.load(f)
    else:
        label_templates = {}
    default_template = {
        "crowCount": 1, "crowAge": 1,
        "alert": False, "begging": False,
        "softSong": False, "rattle": False, "mob": False,
        "quality": 2, "reviewed": False
    }
    cluster_id = STARTING_CLUSTER_ID

    # Process merged clusters.
    for leaf in merged_leaves:
        center = compute_leaf_center(norm_emb, leaf["indices"])
        sorted_indices = sorted(leaf["indices"],
                                key=lambda idx: np.dot(norm_emb[idx], center),
                                reverse=True)
        if PREVIEW_CLUSTERS:
            while True:
                print(f"\nMerged Cluster {cluster_id} (Size {leaf['size']}):")
                for idx in sorted_indices[:PREVIEW_PER_CLUSTER]:
                    file_id, sec = ids[idx]
                    sim = np.dot(norm_emb[idx], center)
                    print(f"  File: {file_id}, Start: {sec}, Similarity: {sim:.4f}")
                    audio_file = dataset_audio_preview_path(file_id)
                    play_audio_preview(audio_file, float(sec), 1.0)
                user_input = input("Press 'R' to repeat or Enter to continue: ")
                if user_input.strip().lower() != 'r':
                    break
        for idx in sorted_indices[:NUM_REPRESENTATIVE]:
            file_id, sec = ids[idx]
            start_time = float(sec)
            end_time = start_time + 1.0
            seg_key = f"{file_id}-{int(start_time)}-{int(end_time)}"
            # Avoid duplicate label keys in new labels.
            if seg_key in new_labels:
                continue
            seg = {"common_name": "American Crow",
                   "scientific_name": "Corvus brachyrhynchos",
                   "start_time": start_time,
                   "end_time": end_time,
                   "confidence": 0.0,
                   "cluster": cluster_id}
            new_segments[file_id].append(seg)
            if str(cluster_id) in label_templates:
                new_labels[seg_key] = label_templates[str(cluster_id)]
            else:
                new_labels[seg_key] = default_template.copy()
            new_labels[seg_key]["cluster"] = cluster_id
        cluster_id += 1

    # Process seed clusters.
    for seed_id, seg_list in seed_clusters.items():
        current_cluster_id = cluster_id
        for seg in seg_list:
            file_id = seg["file_id"]
            start_time = float(seg["start_time"])
            end_time = float(seg["end_time"])
            seg_key = f"{file_id}-{int(start_time)}-{int(end_time)}"
            if seg_key in new_labels:
                continue
            seg_entry = {"common_name": "American Crow",
                         "scientific_name": "Corvus brachyrhynchos",
                         "start_time": start_time,
                         "end_time": end_time,
                         "confidence": 0.0,
                         "cluster": current_cluster_id}
            new_segments[file_id].append(seg_entry)
            if str(cluster_id) in label_templates:
                new_labels[seg_key] = label_templates[str(cluster_id)]
            else:
                new_labels[seg_key] = default_template.copy()
            new_labels[seg_key]["cluster"] = current_cluster_id
        if PREVIEW_SEEDS:
            while True:
                print(f"\nSeed Cluster {current_cluster_id}:")
                for seg in seg_list[:PREVIEW_PER_CLUSTER]:
                    print(f"  File: {seg['file_id']}, Start: {seg['start_time']}, End: {seg['end_time']}")
                    audio_file = dataset_audio_preview_path(seg["file_id"])
                    play_audio_preview(audio_file, seg["start_time"], 1.0)
                user_input = input("Press 'R' to repeat or Enter to continue: ")
                if user_input.strip().lower() != 'r':
                    break
        cluster_id += 1

    # Merge new segments with existing segments.
    for file_id, seg_list in new_segments.items():
        if file_id not in existing_segments:
            existing_segments[file_id] = seg_list
        else:
            # For each new segment, check if a segment with the same start and end exists.
            existing_keys = {(seg["start_time"], seg["end_time"]) for seg in existing_segments[file_id]}
            for seg in seg_list:
                key = (seg["start_time"], seg["end_time"])
                if key not in existing_keys:
                    existing_segments[file_id].append(seg)
                else:
                    print(f"Skipping duplicate segment for file {file_id} at {key}")

    # Merge new labels with existing labels.
    for key, label in new_labels.items():
        if key in existing_labels:
            print(f"Skipping duplicate label: {key}")
        else:
            existing_labels[key] = label

    # Ensure segments in each file are sorted by start_time.
    for file_id in existing_segments:
        existing_segments[file_id] = sorted(existing_segments[file_id], key=lambda x: x["start_time"])

    # Save updated JSON outputs.
    with open(output_segments, "w") as f:
        json.dump(existing_segments, f, indent=2)
    with open(output_labels, "w") as f:
        json.dump(existing_labels, f, indent=2)

    print(f"\nSaved segments to {output_segments}")
    print(f"Saved cluster labels to {output_labels}")
    return existing_segments, existing_labels


def main(dataset_name="all-public", cache_base=None):
    global CURRENT_DATASET, CACHE_BASE
    CURRENT_DATASET = dataset_name
    CACHE_BASE = cache_base
    random.seed(42)
    np.random.seed(42)

    # Load embeddings.
    embeddings, ids, total = load_non_silent_embeddings(dataset_name, VOLUME_THRESHOLD, cache_base=cache_base)
    print(f"Processed {total} seconds. Non-silent segments: {embeddings.shape[0]}")
    embeddings, ids = random_subsample(embeddings, ids, SUBSAMPLE_FACTOR)
    print(f"After subsampling: {embeddings.shape[0]} segments")

    # Dimensionality reduction.
    reduced, _ = reduce_dim_pca(embeddings, PCA_COMPONENTS)

    # Normalize embeddings.
    norm_emb = normalize_embeddings(reduced)

    # Build a Faiss index for similarity search.
    d = norm_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(norm_emb)

    # Hierarchical clustering via recursive k-means splitting.
    if not ONLY_OUTPUT_SEEDS:
        all_indices = list(range(len(norm_emb)))
        hierarchy = hierarchical_split(norm_emb, all_indices, max_size=MAX_CLUSTER_SIZE)
        leaves = collect_leaves(hierarchy)
        print(f"\nTotal leaf clusters: {len(leaves)}")

        # Compute leaf centers and determine merge candidates.
        centers = {i: compute_leaf_center(norm_emb, leaf["indices"]) for i, leaf in enumerate(leaves)}
        num_leaves = len(centers)
        merges = []
        for i in range(num_leaves):
            for j in range(i + 1, num_leaves):
                cosine_distance = 1 - np.dot(centers[i], centers[j])
                if cosine_distance < MERGE_THRESHOLD:
                    merges.append((i, j, cosine_distance))
        merged_leaves = merge_leaves_once(leaves, merges)
        print(f"After merging, total clusters: {len(merged_leaves)}")
    else:
        merged_leaves = []

    # Process seed examples.
    seed_clusters = process_seed_examples(norm_emb, ids, faiss_index, max_per_file=4)
    if seed_clusters:
        print(f"\nProcessed {len(seed_clusters)} seed clusters.")

    # Combine merged and seed clusters, preview as needed, sort and save JSON.
    build_and_save_clusters(merged_leaves, seed_clusters, ids, norm_emb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find candidate training clusters from dataset embeddings.")
    parser.add_argument("--dataset", default="all-public", help="Dataset to cluster.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    args = parser.parse_args()
    main(dataset_name=args.dataset, cache_base=args.cache_dir)
