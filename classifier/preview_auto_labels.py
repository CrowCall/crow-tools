#!/usr/bin/env python3
import os
import json
import librosa
import sounddevice as sd
import argparse
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowtools.datasets import (
    get_dataset_artifact_path,
    load_dataset_auto_labels,
    resolve_dataset_audio_path,
)


def play_audio_preview(audio_file, start_time, duration):
    try:
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=duration)
        sd.play(y, sr)
        sd.wait()
    except Exception as e:
        print(f"Error playing {audio_file}: {e}")


def save_cluster_labels(cluster_labels, cluster_labels_file):
    with open(cluster_labels_file, "w") as f:
        json.dump(cluster_labels, f, indent=4)
    print(f"Progress saved to {cluster_labels_file}.")


def append_missing_segments(cluster_labels_file, cluster_segments_file):
    # Load cluster_labels.
    if os.path.exists(cluster_labels_file):
        with open(cluster_labels_file, "r") as f:
            cluster_labels = json.load(f)
    else:
        print(f"No cluster labels found at {cluster_labels_file}. Skipping appending segments.")
        return

    # Load or initialize cluster_segments.
    if os.path.exists(cluster_segments_file):
        with open(cluster_segments_file, "r") as f:
            cluster_segments = json.load(f)
    else:
        cluster_segments = {}

    # Process each detection in cluster_labels.
    for key, detection in cluster_labels.items():
        parts = key.split("-")
        if len(parts) != 3:
            print(f"Skipping key with unexpected format: {key}")
            continue
        file_id, start_str, end_str = parts
        try:
            start_time = float(start_str)
            end_time = float(end_str)
        except ValueError:
            print(f"Invalid start/end times in key: {key}")
            continue

        # Ensure the file_id exists in cluster_segments.
        if file_id not in cluster_segments:
            cluster_segments[file_id] = []

        # Check if the segment (by start and end times) already exists.
        segment_exists = any(
            seg.get("start_time") == start_time and seg.get("end_time") == end_time
            for seg in cluster_segments[file_id]
        )
        if not segment_exists:
            # Build a new segment entry following the correct format.
            segment_entry = {
                "common_name": "American Crow",
                "scientific_name": "Corvus brachyrhynchos",
                "start_time": start_time,
                "end_time": end_time,
                "confidence": detection.get("confidence", 0),
                "cluster": detection.get("cluster")
            }
            cluster_segments[file_id].append(segment_entry)
            print(f"Appended segment {key} to cluster_segments under file id {file_id}.")

    # Save the updated cluster_segments.
    with open(cluster_segments_file, "w") as f:
        json.dump(cluster_segments, f, indent=4)
    print(f"Updated cluster_segments saved to {cluster_segments_file}.")

def main(attribute, cluster, offset, dataset_name="all-public", cache_base=None):
    cluster_labels_file = get_dataset_artifact_path(dataset_name, "cluster_labels.json", cache_base=cache_base)
    cluster_segments_file = get_dataset_artifact_path(dataset_name, "cluster_segments.json", cache_base=cache_base)

    auto_labels = load_dataset_auto_labels(dataset_name, cache_base=cache_base)

    # Load or initialize cluster_labels.json.
    if os.path.exists(cluster_labels_file):
        with open(cluster_labels_file, "r") as f:
            cluster_labels = json.load(f)
    else:
        cluster_labels = {}

    # Handle filter attribute with optional value.
    if ":" in attribute:
        attr_key, attr_val_str = attribute.split(":", 1)
        if attr_val_str == "":
            attr_val = True
        else:
            try:
                attr_val = int(attr_val_str)
            except ValueError:
                attr_val = attr_val_str
    else:
        attr_key = attribute
        attr_val = True

    # Filter for detections with the specified attribute.
    filtered_keys = [k for k, v in auto_labels.items() if v.get(attr_key) == attr_val]
    print(f"Found {len(filtered_keys)} detections with {attr_key} == {attr_val}.")

    # Set to store file_ids that should be skipped.
    skipped_file_ids = set()

    for key in filtered_keys[offset:]:
        parts = key.split("-")
        if len(parts) != 3:
            print(f"Skipping key with unexpected format: {key}")
            continue
        file_id, start, end = parts

        # Skip detections from file_ids already marked to be skipped.
        if file_id in skipped_file_ids:
            print(f"Skipping detection {key} because file id {file_id} is marked as skipped.")
            continue

        # Skip if already included.
        if key in cluster_labels:
            print(f"Skipped {key}: already in cluster_labels.")
            continue

        detection = auto_labels[key]
        audio_file = resolve_dataset_audio_path(dataset_name, file_id, cache_base=cache_base)
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            continue

        try:
            start_time = float(start)
            end_time = float(end)
        except ValueError:
            print(f"Invalid start/end in key: {key}")
            continue

        duration = end_time - start_time
        print(f"\nPlaying detection {key} from {audio_file} (start: {start_time}, duration: {duration}) ...")
        print(detection)
        play_audio_preview(audio_file, start_time, duration)

        response = input("Include this detection? (Y = include, N = skip, Z = add as garbage, S = skip file, Q = quit): ").strip().lower()
        if response in ("q", "quit"):
            print("Quitting early. Saving progress...")
            save_cluster_labels(cluster_labels, cluster_labels_file)
            append_missing_segments(cluster_labels_file, cluster_segments_file)
            return
        elif response == "s":
            skipped_file_ids.add(file_id)
            print(f"Skipping all further detections for file id {file_id}.")
            continue
        elif response == "y":
            new_detection = detection.copy()
            new_detection["cluster"] = cluster
            cluster_labels[key] = new_detection
            print(f"Added {key} to cluster_labels with cluster {cluster}.")
            save_cluster_labels(cluster_labels, cluster_labels_file)
        elif response == "z":
            # Create a garbage label.
            garbage_detection = {
                "crowCount": 0,
                "crowAge": 1,
                "alert": False,
                "begging": False,
                "softSong": False,
                "rattle": False,
                "mob": False,
                "quality": 1,
                "cluster": cluster
            }
            cluster_labels[key] = garbage_detection
            print(f"Added {key} as garbage to cluster_labels with cluster {cluster}.")
            save_cluster_labels(cluster_labels, cluster_labels_file)
        else:
            print("Skipped.")

    print("\nAll detections processed.")
    save_cluster_labels(cluster_labels, cluster_labels_file)
    append_missing_segments(cluster_labels_file, cluster_segments_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview and promote auto-labeled segments into cluster labels.")
    parser.add_argument("--dataset", default="all-public", help="Dataset to read from and write to.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    args = parser.parse_args()

    attr = input("Enter detection attribute to filter (e.g., rattle, softSong, begging, mob, alert, quality:1, crowCount:2, crowAge:1): ").strip()

    cluster_labels_file = get_dataset_artifact_path(args.dataset, "cluster_labels.json", cache_base=args.cache_dir)
    default_cluster = 1
    if os.path.exists(cluster_labels_file):
        with open(cluster_labels_file, "r") as f:
            existing_cluster_labels = json.load(f)
        cluster_values = []
        for detection in existing_cluster_labels.values():
            cluster_val = detection.get("cluster")
            if isinstance(cluster_val, int):
                cluster_values.append(cluster_val)
        if cluster_values:
            default_cluster = max(cluster_values) + 1

    clus_input = input(f"Enter cluster number to assign [default: {default_cluster}]: ").strip()
    clus = int(clus_input) if clus_input else default_cluster

    offset_input = input("Enter starting offset [default: 0]: ").strip()
    offset = int(offset_input) if offset_input else 0

    main(attr, clus, offset, dataset_name=args.dataset, cache_base=args.cache_dir)
