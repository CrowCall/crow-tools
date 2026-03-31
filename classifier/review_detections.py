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


def save_labels(labels, labels_file):
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=4)
    print(f"Progress saved to {labels_file}.")


def append_missing_segments(labels_file, segments_file):
    # Load labels.
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            labels = json.load(f)
    else:
        print(f"No labels found at {labels_file}. Skipping appending segments.")
        return

    # Load or initialize segments.
    if os.path.exists(segments_file):
        with open(segments_file, "r") as f:
            segments = json.load(f)
    else:
        segments = {}

    # Process each detection in labels.
    for key, detection in labels.items():
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

        # Ensure the file_id exists in segments.
        if file_id not in segments:
            segments[file_id] = []

        # Check if the segment (by start and end times) already exists.
        segment_exists = any(
            seg.get("start_time") == start_time and seg.get("end_time") == end_time
            for seg in segments[file_id]
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
            segments[file_id].append(segment_entry)
            print(f"Appended segment {key} to segments under file id {file_id}.")

    # Save the updated segments.
    with open(segments_file, "w") as f:
        json.dump(segments, f, indent=4)
    print(f"Updated segments saved to {segments_file}.")

def main(attribute, cluster, offset, dataset_name="all-public", cache_base=None):
    labels_file = get_dataset_artifact_path(dataset_name, "labels.json", cache_base=cache_base)
    segments_file = get_dataset_artifact_path(dataset_name, "segments.json", cache_base=cache_base)

    auto_labels = load_dataset_auto_labels(dataset_name, cache_base=cache_base)

    # Load or initialize labels.json.
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            labels = json.load(f)
    else:
        labels = {}

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
        if key in labels:
            print(f"Skipped {key}: already in labels.")
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
            save_labels(labels, labels_file)
            append_missing_segments(labels_file, segments_file)
            return
        elif response == "s":
            skipped_file_ids.add(file_id)
            print(f"Skipping all further detections for file id {file_id}.")
            continue
        elif response == "y":
            new_detection = detection.copy()
            new_detection["cluster"] = cluster
            labels[key] = new_detection
            print(f"Added {key} to labels with cluster {cluster}.")
            save_labels(labels, labels_file)
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
            labels[key] = garbage_detection
            print(f"Added {key} as garbage to labels with cluster {cluster}.")
            save_labels(labels, labels_file)
        else:
            print("Skipped.")

    print("\nAll detections processed.")
    save_labels(labels, labels_file)
    append_missing_segments(labels_file, segments_file)

def get_default_cluster(dataset_name="all-public", cache_base=None):
    labels_file = get_dataset_artifact_path(dataset_name, "labels.json", cache_base=cache_base)
    default_cluster = 1
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            existing_labels = json.load(f)
        cluster_values = []
        for detection in existing_labels.values():
            cluster_val = detection.get("cluster")
            if isinstance(cluster_val, int):
                cluster_values.append(cluster_val)
        if cluster_values:
            default_cluster = max(cluster_values) + 1
    return default_cluster


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Review detected segments and promote accepted items into dataset labels.")
    parser.add_argument("--dataset", default="all-public", help="Dataset to read from and write to.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    parser.add_argument(
        "--attribute",
        default=None,
        help="Detection attribute filter, e.g. rattle, softSong, quality:1, crowCount:2.",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Cluster to assign to accepted detections. Defaults to the next available cluster.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting offset within the filtered detections.",
    )
    args = parser.parse_args(argv)

    attr = args.attribute or input(
        "Enter detection attribute to filter (e.g., rattle, softSong, begging, mob, alert, quality:1, crowCount:2, crowAge:1): "
    ).strip()

    default_cluster = get_default_cluster(args.dataset, args.cache_dir)
    if args.cluster is None:
        clus_input = input(f"Enter cluster number to assign [default: {default_cluster}]: ").strip()
        clus = int(clus_input) if clus_input else default_cluster
    else:
        clus = args.cluster

    main(attr, clus, args.offset, dataset_name=args.dataset, cache_base=args.cache_dir)


if __name__ == "__main__":
    main_cli()
