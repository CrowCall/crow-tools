import os
import json
import csv
from detector.detect import detect_file_segments

# Define paths.
PATH = os.path.dirname(__file__)
public_path = os.path.join(PATH, "..", ".cache")
csv_paths = [
    os.path.join(public_path, "csv", "crows.csv"),
    os.path.join(public_path, "csv", "crows-xeno-canto.csv")
]
library_dir = os.path.join(public_path, "library")
segments_path = os.path.join(public_path, "segments.json")

def start_detections():
    # Load previously processed segments if available.
    if os.path.exists(segments_path):
        with open(segments_path, "r") as f:
            segments = json.load(f)
    else:
        segments = {}

    # Using CSV files to get unique file IDs.
    processed_ids = set(segments.keys())  # already processed file IDs
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row["ML Catalog Number"]
                if file_id in processed_ids:
                    print(f"Skipping file {file_id} (already processed)")
                    continue
                processed_ids.add(file_id)
                detections = detect_file_segments(file_id, public_path=public_path)
                if detections:
                    segments[file_id] = detections
                    # Count unique boolean features in this file.
                    features_to_report = ["alert", "mob", "begging", "softSong", "rattle"]
                    feature_counts = {feat: sum(1 for det in detections if det.get(feat, False))
                                      for feat in features_to_report}
                    feature_counts_str = ", ".join(f"{feat}: {count}"
                                                   for feat, count in feature_counts.items() if count > 0)
                    print(f">>>>> Found {len(detections)} detections in file {file_id} ({feature_counts_str})")
                else:
                    print(f"<<<<<< No detections in file {file_id}")

    # Save all segments.
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"***** Saved segments for {len(segments)} files to {segments_path}")

    # Recalculate summary statistics from the segments data.
    files_with_detections = 0
    total_detections = 0

    # For binary attributes, we sum up the counts.
    binary_totals = {
        "alert": 0,
        "begging": 0,
        "softSong": 0,
        "rattle": 0,
        "mob": 0,
    }

    # For non-binary attributes, we group counts by each unique value.
    grouped_counts = {
        "crowCount": {},
        "crowAge": {},
        "quality": {}
    }

    for file_id, det_list in segments.items():
        if det_list:
            files_with_detections += 1
            total_detections += len(det_list)
            for det in det_list:
                # Update binary attributes.
                binary_totals["alert"] += int(det.get("alert", False))
                binary_totals["begging"] += int(det.get("begging", False))
                binary_totals["softSong"] += int(det.get("softSong", False))
                binary_totals["rattle"] += int(det.get("rattle", False))
                binary_totals["mob"] += int(det.get("mob", False))
                # Update non-binary grouped counts.
                for attr in ["crowCount", "crowAge", "quality"]:
                    value = det.get(attr)
                    if value is None:
                        continue
                    # Use the value as-is (no additional +1).
                    grouped_counts[attr][value] = grouped_counts[attr].get(value, 0) + 1

    # Print summary.
    print("\n===== Detection Summary =====")
    print(f"Files with detections: {files_with_detections}")
    if files_with_detections > 0:
        print(f"Average detections per file: {total_detections / files_with_detections:.2f}")
    print(f"Total detections: {total_detections}")

    print("\nTotals for non-binary attributes:")
    for attr, counts in grouped_counts.items():
        counts_str = ", ".join(f"{val} = {counts[val]}" for val in sorted(counts.keys()))
        print(f"  {attr}: {counts_str}")

    print("\nTotals for binary attributes:")
    for attr, total in binary_totals.items():
        print(f"  {attr}: {total}")

if __name__ == "__main__":
    start_detections()
