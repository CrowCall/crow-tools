import os
import glob
import json
import random
import numpy as np
import librosa
import sounddevice as sd
import pandas as pd
from sklearn.decomposition import PCA
import faiss
from collections import defaultdict

# Paths
BASE_PATH = os.path.dirname(__file__)
EMBEDDINGS_DIR = os.path.join(BASE_PATH, "..", ".cache", "embeddings-denoised-1-sec")
VOLUMES_DIR = os.path.join(BASE_PATH, "..", ".cache", "embeddings-denoised-volumes")
OUTPUT_SEGMENTS = os.path.join(BASE_PATH, "..", ".cache", "cluster_segments.json")
OUTPUT_LABELS = os.path.join(BASE_PATH, "..", ".cache", "cluster_labels.json")
AUDIO_DIR = os.path.join(BASE_PATH, "..", ".cache", "library-denoised")
INDEX_PATH = os.path.join(BASE_PATH, "..", ".cache", "faiss_index.index")
LABEL_TEMPLATE_FILE = os.path.join(BASE_PATH, "..", ".cache", "cluster_segments_labels.json")

# Parameters
VOLUME_THRESHOLD = 0.0005
SUBSAMPLE_FACTOR = 0.1
PCA_COMPONENTS = 64
MAX_CLUSTER_SIZE = 500  # Maximum leaf size before splitting stops.
MERGE_THRESHOLD = 0.15  # Merge leaves if cosine distance < 0.15 (single pass)
PREVIEW = False
PREVIEW_PER_CLUSTER = 10
NUM_REPRESENTATIVE = 30  # Number of segments per merged cluster for labeling

# Seed examples for new clusters.
SEED_EXAMPLES = [
    {"file_id": "408950861", "start": 20.0, "end": 21.0},
    {"file_id": "13123", "start": 34.0, "end": 35.0},

]


def play_audio_preview(audio_file, start_time, duration):
    try:
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)
        sd.play(y, sr)
        sd.wait()
    except Exception as e:
        print(f"Error playing {audio_file}: {e}")


def load_non_silent_embeddings(emb_dir, vol_dir, thresh):
    emb_list, ids, total = [], [], 0
    for path in sorted(glob.glob(os.path.join(emb_dir, "*.npy"))):
        file_id = os.path.splitext(os.path.basename(path))[0]
        vol_path = os.path.join(vol_dir, f"{file_id}.npy")
        if not os.path.exists(vol_path):
            continue
        try:
            emb_data = np.load(path)
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
    try:
        seed_file_ids = {seed["file_id"] for seed in SEED_EXAMPLES}
    except NameError:
        seed_file_ids = set()
    # Get indices for embeddings with a file_id in the seed set.
    seed_indices = [i for i, (file_id, _) in enumerate(ids) if file_id in seed_file_ids]
    # All other indices.
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


def print_hierarchy_summary(tree, indent=0):
    prefix = "  " * indent
    if "children" in tree:
        print(f"{prefix}Node (Level {tree['level']}) - Size: {tree['size']}")
        for child in tree["children"].values():
            print_hierarchy_summary(child, indent + 1)
    else:
        print(f"{prefix}Leaf (Level {tree['level']}) - Size: {tree['size']}")


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


def compute_leaf_centers(leaves, norm_emb):
    centers = {}
    for i, leaf in enumerate(leaves):
        centers[i] = compute_leaf_center(norm_emb, leaf["indices"])
    return centers


def compute_leaf_similarities(centers):
    num_leaves = len(centers)
    sim_matrix = np.zeros((num_leaves, num_leaves))
    for i in range(num_leaves):
        for j in range(num_leaves):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = np.dot(centers[i], centers[j])
    return sim_matrix


def aggregate_leaf_merges(sim_matrix, threshold=0.15):
    num_leaves = sim_matrix.shape[0]
    merges = []
    for i in range(num_leaves):
        for j in range(i + 1, num_leaves):
            cosine_distance = 1 - sim_matrix[i, j]
            if cosine_distance < threshold:
                merges.append((i, j, cosine_distance))
    return merges


def print_leaf_merge_table(merges):
    print("\nLeaf Merge Candidates (Single Pass):")
    print("Leaf  |  Merged With (Cosine Distance)")
    print("--------------------------------------")
    for i, j, dist in merges:
        print(f"  {i}   |   {j} ({dist:.2f})")


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


def print_leaf_similarity_table(sim_matrix, top_n=3):
    num_leaves = sim_matrix.shape[0]
    print("\nMerged Leaf Similarity Table:")
    for i in range(num_leaves):
        sims = []
        for j in range(num_leaves):
            if i != j:
                sims.append((j, 1 - sim_matrix[i, j]))
        sims.sort(key=lambda x: x[1])
        top = sims[:top_n]
        row = f"Leaf {i} | " + " | ".join(f"{j} ({dist:.2f})" for j, dist in top)
        print(row)


def build_cluster_segments_and_labels(merged_leaves, ids, norm_emb, num_representative=30):
    segments = defaultdict(list)
    labels = {}
    if os.path.exists(LABEL_TEMPLATE_FILE):
        with open(LABEL_TEMPLATE_FILE, "r") as f:
            label_templates = json.load(f)
    else:
        label_templates = {}
    default_template = {
        "crowCount": 1, "crowAge": 1,
        "alert": False, "begging": False, "grief": False,
        "softSong": False, "rattle": False, "mob": False,
        "quality": 2, "reviewed": False
    }
    # For each merged leaf, sort items by similarity to the leaf center.
    for cluster_id, leaf in enumerate(merged_leaves, start=1):
        center = compute_leaf_center(norm_emb, leaf["indices"])
        sorted_indices = sorted(leaf["indices"], key=lambda idx: np.dot(norm_emb[idx], center), reverse=True)
        for idx in sorted_indices[:num_representative]:
            file_id, sec = ids[idx]
            seg = {
                "common_name": "American Crow",
                "scientific_name": "Corvus brachyrhynchos",
                "start_time": float(sec),
                "end_time": float(sec + 1),
                "confidence": 0.0,
                "cluster": cluster_id
            }
            segments[file_id].append(seg)
            seg_key = f"{file_id}-{sec}-{sec + 1}"
            if str(cluster_id) in label_templates:
                labels[seg_key] = label_templates[str(cluster_id)]
            else:
                labels[seg_key] = default_template.copy()
            labels[seg_key]["cluster"] = cluster_id
    # Sort segments in each file by start_time.
    for file in segments:
        segments[file].sort(key=lambda x: x["start_time"])
    return dict(segments), labels


def process_seed_example(seed, norm_emb, ids, faiss_index, max_per_file=4):
    # For the given seed example, find similar segments via Faiss.
    target_index = None
    for i, (file_id, sec) in enumerate(ids):
        if file_id == seed["file_id"] and int(sec) == int(seed["start"]):
            target_index = i
            break
    if target_index is None:
        print(f"Seed example {seed} not found.")
        return None
    vec = norm_emb[target_index].reshape(1, norm_emb.shape[1])
    D, I = faiss_index.search(vec, NUM_REPRESENTATIVE)
    results = []
    for idx, sim in zip(I[0], D[0]):
        results.append((ids[idx][0], ids[idx][1], 1 - sim))
    # Limit number of results per file.
    file_counts = defaultdict(int)
    filtered = []
    for file_id, sec, dist in results:
        if file_counts[file_id] < max_per_file:
            filtered.append((file_id, sec, dist))
            file_counts[file_id] += 1
    return filtered


def process_seed_examples(SEED_EXAMPLES, norm_emb, ids, faiss_index, max_per_file=4):
    seed_clusters = {}
    seed_labels = {}
    default_template = {
        "crowCount": 1, "crowAge": 1,
        "alert": False, "begging": False, "grief": False,
        "softSong": False, "rattle": False, "mob": False,
        "quality": 2, "reviewed": False
    }
    for i, seed in enumerate(SEED_EXAMPLES):
        similar = process_seed_example(seed, norm_emb, ids, faiss_index, max_per_file)
        if similar is not None:
            seed_clusters[str(i)] = [{"file_id": f, "start_time": float(sec), "end_time": float(sec + 1)} for
                                     f, sec, dist in similar]
            # Use the seed cluster id to label each representative segment.
            for f, sec, dist in similar:
                key = f"{f}-{sec}-{sec + 1}"
                seed_labels[key] = default_template.copy()
            # Optionally, preview the seed cluster.
            if PREVIEW:
                print(f"\nSeed Cluster {i} from seed example {seed}:")
                for f, sec, dist in similar[:PREVIEW_PER_CLUSTER]:
                    print(f"  File: {f}, Start: {sec}, Cosine Distance: {dist:.2f}")
                    audio_file = os.path.join(AUDIO_DIR, f"{f}.wav")
                    play_audio_preview(audio_file, float(sec), 1.0)
    return seed_clusters, seed_labels


def main():
    random.seed(42)
    np.random.seed(42)

    # Load embeddings.
    embeddings, ids, total = load_non_silent_embeddings(EMBEDDINGS_DIR, VOLUMES_DIR, VOLUME_THRESHOLD)
    print(f"Processed {total} seconds. Non-silent segments: {embeddings.shape[0]}")
    embeddings, ids = random_subsample(embeddings, ids, SUBSAMPLE_FACTOR)
    print(f"After subsampling: {embeddings.shape[0]} segments")

    # Dimensionality reduction.
    reduced, pca = reduce_dim_pca(embeddings, PCA_COMPONENTS)

    # Normalize embeddings.
    norm_emb = normalize_embeddings(reduced)

    # Build a Faiss index for similarity search.
    d = norm_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(norm_emb)

    # Recursive k-means splitting.
    all_indices = list(range(len(norm_emb)))
    hierarchy = hierarchical_split(norm_emb, all_indices, max_size=MAX_CLUSTER_SIZE)

    print("\nHierarchical Clustering Summary (Tree):")
    print_hierarchy_summary(hierarchy)
    leaves = collect_leaves(hierarchy)
    sizes = [leaf["size"] for leaf in leaves]
    print(f"\nTotal leaf clusters: {len(leaves)}")
    print(f"Leaf size distribution: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # Compute leaf centers and merge leaves in one pass.
    centers = {i: compute_leaf_center(norm_emb, leaf["indices"]) for i, leaf in enumerate(leaves)}
    num_leaves = len(centers)
    sim_matrix = np.zeros((num_leaves, num_leaves))
    for i in range(num_leaves):
        for j in range(num_leaves):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = np.dot(centers[i], centers[j])
    merges = []
    for i in range(num_leaves):
        for j in range(i + 1, num_leaves):
            cosine_distance = 1 - sim_matrix[i, j]
            if cosine_distance < MERGE_THRESHOLD:
                merges.append((i, j, cosine_distance))
    print_leaf_merge_table(merges)
    merged_leaves = merge_leaves_once(leaves, merges)
    print(f"\nAfter merging, total clusters: {len(merged_leaves)}")
    merged_sizes = [leaf["size"] for leaf in merged_leaves]
    print(f"Merged leaf size distribution: min={min(merged_sizes)}, max={max(merged_sizes)}, mean={np.mean(merged_sizes):.1f}")

    # Print merged leaf similarity table.
    merged_centers = {i: compute_leaf_center(norm_emb, leaf["indices"]) for i, leaf in enumerate(merged_leaves)}
    merged_sim_matrix = np.zeros((len(merged_centers), len(merged_centers)))
    for i in range(len(merged_centers)):
        for j in range(len(merged_centers)):
            if i == j:
                merged_sim_matrix[i, j] = 1.0
            else:
                merged_sim_matrix[i, j] = np.dot(merged_centers[i], merged_centers[j])
    print("\nMerged Leaf Similarity Table:")
    for i in range(len(merged_centers)):
        sims = []
        for j in range(len(merged_centers)):
            if i != j:
                sims.append((j, 1 - merged_sim_matrix[i, j]))
        sims.sort(key=lambda x: x[1])
        top = sims[:3]
        row = f"Leaf {i} | " + " | ".join(f"{j} ({dist:.2f})" for j, dist in top)
        print(row)

    # Preview merged clusters.
    # if PREVIEW:
    #     print("\nPreviewing merged clusters (sorted by similarity to merged leaf center):")
    #     for i, leaf in enumerate(merged_leaves):
    #         center = compute_leaf_center(norm_emb, leaf["indices"])
    #         sorted_indices = sorted(leaf["indices"], key=lambda idx: np.dot(norm_emb[idx], center), reverse=True)
    #         print(f"\nMerged Cluster {i} (Size {leaf['size']}):")
    #         for idx in sorted_indices[:PREVIEW_PER_CLUSTER]:
    #             file_id, sec = ids[idx]
    #             sim = np.dot(norm_emb[idx], center)
    #             print(f"  File: {file_id}, Start: {sec}, Cosine Similarity to Merged Center: {sim:.4f}")
    #             audio_file = os.path.join(AUDIO_DIR, f"{file_id}.wav")
    #             play_audio_preview(audio_file, float(sec), 1.0)

    # Process seed examples.
    seed_clusters, seed_labels = process_seed_examples(SEED_EXAMPLES, norm_emb, ids, faiss_index, max_per_file=4)
    if seed_clusters:
        print(f"\nProcessed {len(seed_clusters)} seed clusters.")

    # Build final segments mapping and cluster labels from merged clusters.
    segments, cluster_labels = build_cluster_segments_and_labels(merged_leaves, ids, norm_emb,
                                                                 num_representative=NUM_REPRESENTATIVE)
    # Append seed clusters into segments and labels as additional clusters.
    for cluster_offset, (key, segs_list) in enumerate(seed_clusters.items(), start=1):
        # Use a new cluster id that is not used in merged clusters.
        new_cluster_id = str(len(merged_leaves) + cluster_offset)
        # Create a new key for each segment.
        for seg in segs_list:
            file_id = seg["file_id"]
            # Append to segments: ensure segments are grouped by file.
            segments[file_id] = segments.get(file_id, []) + [{
                "common_name": "American Crow",
                "scientific_name": "Corvus brachyrhynchos",
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "cluster": new_cluster_id
            }]
            seg_key = f"{file_id}-{seg['start_time']}-{seg['end_time']}"
            seed_labels[seg_key] = seed_labels.get(seg_key, {
                "crowCount": 1, "crowAge": 1,
                "alert": False, "begging": False, "grief": False,
                "softSong": False, "rattle": False, "mob": False,
                "quality": 2, "reviewed": False
            })
        # Also add this new cluster to cluster_labels.
        cluster_labels[new_cluster_id] = seed_labels
    # Save final outputs.
    with open(OUTPUT_SEGMENTS, "w") as f:
        json.dump(segments, f, indent=2)
    with open(OUTPUT_LABELS, "w") as f:
        json.dump(cluster_labels, f, indent=2)
    print(f"\nSaved final segments mapping to {OUTPUT_SEGMENTS}")
    print(f"Saved final cluster labels to {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()
