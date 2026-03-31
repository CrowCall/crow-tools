#!/usr/bin/env python3
import json
import os
import numpy as np
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowtools.datasets import get_dataset_artifact_path

# Adjust CACHE_DIR to point to your .cache folder.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
DATASET_NAME = "all-public"
SEGMENTS_FILE = get_dataset_artifact_path(DATASET_NAME, "segments.json", cache_base=CACHE_DIR)
LABELS_FILE = get_dataset_artifact_path(DATASET_NAME, "labels.json", cache_base=CACHE_DIR)

def remove_duplicate_segments(segments_data):
    """
    For each file_id in segments_data, remove duplicate segments by:
      1. Comparing start and end times.
      2. Checking that the embedding slice lengths (using .shape[0]) are identical.
      3. Comparing the embedding arrays with near-equality.
    Only one segment (and its corresponding label) is kept per unique embedding.
    """
    deduped = {}
    seen = {}
    for file_id, segments in segments_data.items():
        unique_segments = []
        emb_path = os.path.join(CACHE_DIR, "embeddings", f"{file_id}.npy")
        try:
            embedding_array = np.load(emb_path)
        except Exception as e:
            print(f"Error loading embedding for {file_id}: {e}")
            embedding_array = None

        # Dictionary grouping segments by (start, end, emb_length)
        for seg in segments:
            start = seg.get("start_time")
            end = seg.get("end_time")
            if embedding_array is not None:
                try:
                    s = int(start)
                    e = int(end)
                    seg_embedding = embedding_array[s:e]
                    emb_length = seg_embedding.shape[0]
                except Exception as ex:
                    print(f"Error processing embedding slice for {file_id} segment {seg}: {ex}")
                    seg_embedding = None
                    emb_length = None
            else:
                seg_embedding = None
                emb_length = None

            composite_key = (start, end, emb_length)
            duplicate_found = False
            if composite_key in seen:
                for existing_embedding in seen[composite_key]:
                    # Use near-equality check with a small tolerance.
                    if seg_embedding is not None and np.allclose(seg_embedding, existing_embedding, atol=1e-5):
                        print(f"Removing duplicate segment in {file_id}: {seg}")
                        duplicate_found = True
                        break
            if duplicate_found:
                continue
            seen.setdefault(composite_key, []).append(seg_embedding)
            unique_segments.append(seg)
        deduped[file_id] = unique_segments
    return deduped

def remove_duplicate_labels(labels_data):
    """
    Remove duplicate label entries from a labels dictionary.
    The keys are expected to be strings of the form "file_id-start-end".
    """
    deduped = {}
    for key, label in labels_data.items():
        try:
            parts = key.split("-")
            if len(parts) != 3:
                print(f"Skipping label with unexpected key format: {key}")
                continue
            file_id, start, end = parts
            key_tuple = (file_id, float(start), float(end))
        except Exception as e:
            print(f"Error parsing key {key}: {e}")
            continue

        if key_tuple in deduped:
            print(f"Removing duplicate label: {key} -> {label}")
        else:
            deduped[key_tuple] = label

    new_labels = {}
    for key_tuple, label in deduped.items():
        new_key = f"{key_tuple[0]}-{int(key_tuple[1])}-{int(key_tuple[2])}"
        new_labels[new_key] = label
    return new_labels

def remove_missing_labels(labels_data, segments_data):
    """
    Remove any label entry for which there is no corresponding segment.
    A corresponding segment is defined as one in segments_data with the same file_id,
    start_time, and end_time.
    """
    new_labels = {}
    for key, label in labels_data.items():
        try:
            file_id, start, end = key.split("-")
            start = float(start)
            end = float(end)
        except Exception as e:
            print(f"Error parsing label key {key}: {e}")
            continue

        if file_id not in segments_data:
            print(f"Removing label {key}: file_id {file_id} not found in segments.")
            continue

        segments = segments_data[file_id]
        found = any(seg.get("start_time") == start and seg.get("end_time") == end for seg in segments)
        if not found:
            print(f"Removing label {key}: corresponding segment not found.")
        else:
            new_labels[key] = label
    return new_labels

def main():
    # Load segments file.
    try:
        with open(SEGMENTS_FILE, "r") as f:
            segments_data = json.load(f)
    except Exception as e:
        print(f"Error loading segments file: {e}")
        return

    # Remove duplicates from segments.
    segments_data = remove_duplicate_segments(segments_data)
    with open(SEGMENTS_FILE, "w") as f:
        json.dump(segments_data, f, indent=4)
    print("Updated segments file.")

    # Load labels file.
    try:
        with open(LABELS_FILE, "r") as f:
            labels_data = json.load(f)
    except Exception as e:
        print(f"Error loading labels file: {e}")
        return

    # Remove duplicate labels.
    labels_data = remove_duplicate_labels(labels_data)
    # Remove label entries for missing segments.
    labels_data = remove_missing_labels(labels_data, segments_data)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels_data, f, indent=4)
    print("Updated labels file.")

if __name__ == "__main__":
    main()
