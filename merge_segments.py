import json
import os
from tqdm import tqdm


def merge_segments(detections, gap_threshold=1.0, conf_threshold=0.15, max_length=15.0):
    """
    Merge adjacent detections if the gap between segments is less than or equal
    to gap_threshold seconds, the confidence difference is less than or equal
    to conf_threshold, and the merged segment length does not exceed max_length seconds.
    The merged segment's confidence is the average of the merged segments.

    :param detections: List of dictionaries. Each should have "start_time", "end_time", "confidence".
    :param gap_threshold: Maximum allowed gap (in seconds) between segments to consider merging.
    :param conf_threshold: Maximum allowed difference in confidence between segments.
    :param max_length: Maximum allowed length (in seconds) for a merged segment.
    :return: A list of merged detections.
    """
    if not detections:
        return []

    # Sort detections by start time
    detections.sort(key=lambda d: d["start_time"])
    merged = []
    # Start with a copy of the first detection
    current = detections[0].copy()
    count = 1
    cum_conf = current.get("confidence", 0.0)

    for det in detections[1:]:
        gap = det["start_time"] - current["end_time"]
        avg_conf = cum_conf / count
        conf_diff = abs(det["confidence"] - avg_conf)
        new_length = det["end_time"] - current["start_time"]

        if gap <= gap_threshold and conf_diff <= conf_threshold and new_length <= max_length:
            # Merge: extend the current segment
            current["end_time"] = det["end_time"]
            cum_conf += det["confidence"]
            count += 1
        else:
            # Finalize current segment with averaged confidence
            current["confidence"] = cum_conf / count
            merged.append(current)
            # Start a new segment with the current detection
            current = det.copy()
            count = 1
            cum_conf = current.get("confidence", 0.0)

    # Append the final segment
    current["confidence"] = cum_conf / count
    merged.append(current)
    return merged


def main():
    segments_path = os.path.join("labeler-vue", "public", "segments.json")

    if not os.path.exists(segments_path):
        print(f"Error: {segments_path} does not exist.")
        return

    with open(segments_path, "r") as f:
        segments = json.load(f)

    # Process each catalog_number's detection list
    merged_segments = {}
    for catalog_number in tqdm(segments, desc="Merging segments"):
        detections = segments[catalog_number]
        merged = merge_segments(detections, gap_threshold=1.0, conf_threshold=0.15, max_length=15.0)
        merged_segments[catalog_number] = merged

    # Save the merged segments back to segments.json
    with open(segments_path, "w") as f:
        json.dump(merged_segments, f, indent=2)
    print(f"Saved merged segments to {segments_path}")


if __name__ == "__main__":
    main()
