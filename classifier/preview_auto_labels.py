#!/usr/bin/env python3
import os
import json
import librosa
import sounddevice as sd


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


def main(attribute, cluster, offset):
    base_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
    auto_labels_file = os.path.join(base_dir, "auto_labels.json")
    cluster_labels_file = os.path.join(base_dir, "cluster_labels.json")
    library_dir = os.path.join(base_dir, "library")

    # Load auto_labels.json.
    with open(auto_labels_file, "r") as f:
        auto_labels = json.load(f)

    # Load or initialize cluster_labels.json.
    if os.path.exists(cluster_labels_file):
        with open(cluster_labels_file, "r") as f:
            cluster_labels = json.load(f)
    else:
        cluster_labels = {}

    # Filter for detections with the specified attribute set to True.
    filtered_keys = [k for k, v in auto_labels.items() if v.get(attribute, False)]
    print(f"Found {len(filtered_keys)} detections with {attribute}=True.")

    for key in filtered_keys[offset:]:
        # Skip if already included.
        if key in cluster_labels:
            print(f"Skipped {key}: already in cluster_labels.")
            continue

        detection = auto_labels[key]
        parts = key.split("-")
        if len(parts) != 3:
            print(f"Skipping key with unexpected format: {key}")
            continue

        file_id, start, end = parts
        audio_file = os.path.join(library_dir, f"{file_id}.mp3")
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

        response = input(
            "Include this detection? (Y = include, N = skip, Z = add as garbage, Q = quit): ").strip().lower()
        if response in ("q", "quit"):
            print("Quitting early. Saving progress...")
            save_cluster_labels(cluster_labels, cluster_labels_file)
            return
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


if __name__ == "__main__":
    attr = input("Enter detection attribute to filter (e.g., rattle, softSong, begging, mob, alert): ").strip()
    clus = input("Enter cluster number to assign: ").strip()
    offset = input("Enter starting offset: ").strip()
    main(attr, int(clus), int(offset))
