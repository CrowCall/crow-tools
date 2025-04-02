#!/usr/bin/env python3
import json
import os

# Adjust CACHE_DIR to point to your .cache folder.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
SEGMENTS_FILE = os.path.join(CACHE_DIR, "cluster_segments.json")
LABELS_FILE = os.path.join(CACHE_DIR, "cluster_labels.json")


def remove_duplicate_segments(segments_data):
    """
    For each file_id in segments_data, remove duplicate segments (based on same start and end).
    """
    deduped = {}
    for file_id, segments in segments_data.items():
        seen = set()
        unique_segments = []
        for seg in segments:
            key = (seg.get("start_time"), seg.get("end_time"))
            if key in seen:
                print(f"Removing duplicate segment in {file_id}: {seg}")
            else:
                seen.add(key)
                unique_segments.append(seg)
        deduped[file_id] = unique_segments
    return deduped


def remove_duplicate_labels(labels_data):
    """
    Remove duplicate label entries from a labels dictionary. The keys are expected to be strings
    of the form "file_id-start-end". If duplicate keys (i.e. same file_id, start, and end) are found,
    only one is kept (the first encountered), and duplicates are printed.
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

    # Rebuild dictionary with keys in the original "file_id-start-end" string format.
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
