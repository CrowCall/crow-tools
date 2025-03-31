#!/usr/bin/env python3
import os
import json
import csv
import numpy as np
from detector.detect import detect_file_segments
from classifier.classify import predict_embedding

# Define paths.
PATH = os.path.dirname(__file__)
public_path = os.path.join(PATH, "..", ".cache")
csv_paths = [
    os.path.join(public_path, "csv", "crows.csv"),
    os.path.join(public_path, "csv", "crows-xeno-canto.csv")
]
library_dir = os.path.join(public_path, "library")
segments_path = os.path.join(public_path, "segments.json")
auto_labels_path = os.path.join(public_path, "auto_labels.json")

def compute_contiguous_stats(segments, auto_labels, target_crowCount, tolerance=0.01):
    """
    Compute contiguous group stats for segments whose auto_labels have a crowCount equal to target_crowCount.
    Contiguity is defined as int(curr["start_time"]) == int(prev["end_time"]).
    Returns a histogram (dict) where keys are the group sizes.
    """
    contiguous_group_hist = {}
    for file_id, seg_list in segments.items():
        # Filter segments by target crowCount.
        valid_segs = []
        for seg in seg_list:
            seg_key = f"{file_id}-{int(seg['start_time'])}-{int(seg['end_time'])}"
            label = auto_labels.get(seg_key)
            if label and label.get("crowCount") == target_crowCount:
                valid_segs.append(seg)
        if not valid_segs:
            continue

        # Sort valid segments by start_time.
        sorted_segs = sorted(valid_segs, key=lambda x: x["start_time"])
        groups = []
        current_group_size = 1

        for i in range(1, len(sorted_segs)):
            prev = sorted_segs[i - 1]
            curr = sorted_segs[i]
            # Using integer conversion as in the segment key.
            if int(curr["start_time"]) == int(prev["end_time"]):
                current_group_size += 1
            else:
                groups.append(current_group_size)
                current_group_size = 1
        groups.append(current_group_size)  # add the final group

        # Update histogram counts.
        for group_size in groups:
            contiguous_group_hist[group_size] = contiguous_group_hist.get(group_size, 0) + 1

    return contiguous_group_hist

def start_detections():
    # Load previously processed auto labels if available.
    if os.path.exists(auto_labels_path):
        with open(auto_labels_path, "r", encoding="utf-8") as f:
            auto_labels = json.load(f)
    else:
        auto_labels = {}

    # Load previously processed segments if available.
    if os.path.exists(segments_path):
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = {}

    # Use keys from auto_labels as already processed file IDs.
    processed_ids = set(auto_labels.keys())
    # (Since auto_labels keys are segment keys, we infer processed file IDs
    #  by collecting the file_id part from each key.)
    processed_file_ids = {key.split("-")[0] for key in processed_ids}

    # Iterate through CSVs for unique file IDs.
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row["ML Catalog Number"]
                if file_id in processed_file_ids:
                    print(f"Skipping file {file_id} (already processed)")
                    continue

                # Run detection for this file.
                detections = detect_file_segments(file_id, public_path=public_path)
                if detections:
                    # For segments, we create a minimal dictionary per detection.
                    segments[file_id] = []
                    # Load the embeddings for the file.
                    embedding_file = os.path.join(public_path, "embeddings-denoised", f"{file_id}.npy")
                    if not os.path.exists(embedding_file):
                        print(f"Embedding file {embedding_file} not found for file {file_id}, skipping.")
                        continue
                    embeddings_array = np.load(embedding_file)

                    for det in detections:
                        st = det.get("start_time")
                        et = det.get("end_time")
                        # Construct a segment key (e.g., "fileid-44-45")
                        segment_key = f"{file_id}-{int(st)}-{int(et)}"
                        # Save minimal segment info.
                        seg_min = {
                            "start_time": st,
                            "end_time": et,
                            "confidence": 0.0,
                            "cluster": 0
                        }
                        segments[file_id].append(seg_min)

                        # Get the embedding corresponding to this segment.
                        idx = int(st)
                        if idx < 0 or idx >= embeddings_array.shape[0]:
                            print(f"Index {idx} out of range for file {file_id}, skipping segment {segment_key}.")
                            continue
                        emb = embeddings_array[idx]
                        # Predict the label.
                        predicted_label = predict_embedding(emb)
                        auto_labels[segment_key] = predicted_label

                    # Optional: print summary for this file.
                    features_to_report = ["alert", "mob", "begging", "softSong", "rattle"]
                    feature_counts = {feat: sum(1 for det in detections if det.get(feat, False))
                                      for feat in features_to_report}
                    feature_counts_str = ", ".join(f"{feat}: {count}"
                                                   for feat, count in feature_counts.items() if count > 0)
                    print(f">>>>> Found {len(detections)} detections in file {file_id} ({feature_counts_str})")
                else:
                    print(f"<<<<<< No detections in file {file_id}")

    # Save segments and auto labels.
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"***** Saved segments for {len(segments)} files to {segments_path}")

    with open(auto_labels_path, "w", encoding="utf-8") as f:
        json.dump(auto_labels, f, indent=4)
    print(f"***** Saved auto labels for {len(auto_labels)} segments to {auto_labels_path}")

    # Recalculate summary statistics from auto_labels.
    total_detections = 0
    binary_totals = {
        "alert": 0,
        "begging": 0,
        "softSong": 0,
        "rattle": 0,
        "mob": 0,
    }

    # For non-binary attributes, group counts by unique value.
    grouped_counts = {
        "crowCount": {},
        "crowAge": {},
        "quality": {}
    }

    for segment_key, label in auto_labels.items():
        total_detections += 1
        # Update binary attributes.
        binary_totals["alert"] += int(label.get("alert", False))
        binary_totals["begging"] += int(label.get("begging", False))
        binary_totals["softSong"] += int(label.get("softSong", False))
        binary_totals["rattle"] += int(label.get("rattle", False))
        binary_totals["mob"] += int(label.get("mob", False))
        # Update non-binary grouped counts.
        for attr in ["crowCount", "crowAge", "quality"]:
            value = label.get(attr)
            if value is None:
                continue
            grouped_counts[attr][value] = grouped_counts[attr].get(value, 0) + 1

    # For total detection time, each detection is one second.
    hours = total_detections // 3600
    minutes = (total_detections % 3600) // 60

    print("\n===== Detection Summary =====")
    print(f"Total detections: {total_detections} ({int(hours):02d}:{int(minutes):02d} HH:MM)")
    print("\nTotals for non-binary attributes:")
    for attr, counts in grouped_counts.items():
        counts_str = ", ".join(f"{val} = {counts[val]}" for val in sorted(counts.keys()))
        print(f"  {attr}: {counts_str}")
    print("\nTotals for binary attributes:")
    for attr, total in binary_totals.items():
        hours = total // 3600
        minutes = (total % 3600) // 60
        print(f"  {attr}: {total} ({int(hours):02d}:{int(minutes):02d} HH:MM)")

    # --- New code: Compute contiguous segments group stats for different crowCount filters ---
    for target in [1, 2, 4]:
        stats = compute_contiguous_stats(segments, auto_labels, target_crowCount=target)
        print(f"\nContiguous Segments Groups (crowCount=={target}):")
        for group_size in sorted(stats.keys())[:10]:
            seg_label = "segment" if group_size == 1 else "segments"
            total_time = stats[group_size]
            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            print(f"  {group_size} {seg_label}: {total_time} ({int(hours):02d}:{int(minutes):02d} HH:MM)")


if __name__ == "__main__":
    start_detections()
