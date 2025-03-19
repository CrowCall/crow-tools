import json


def remove_long_segments(file_path, max_duration=3.0):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_before = 0
    total_after = 0
    summary = {}

    # Loop over each key and filter segments, counting before and after
    for key, segments in data.items():
        count_before = len(segments)
        total_before += count_before

        # Filter segments by duration
        filtered_segments = [
            seg for seg in segments
            if (seg['end_time'] - seg['start_time']) <= max_duration
        ]
        count_after = len(filtered_segments)
        total_after += count_after

        summary[key] = {"before": count_before, "after": count_after}
        data[key] = filtered_segments

    # Write the updated data back to the same file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Print a nice summary
    print("Segment Removal Summary:")
    for key, counts in summary.items():
        print(f"Key {key}: {counts['before']} segments originally, {counts['after']} segments after filtering.")
    print("-" * 50)
    print(f"Total segments before filtering: {total_before}")
    print(f"Total segments after filtering:  {total_after}")
    print(f"Total segments removed:         {total_before - total_after}")


if __name__ == "__main__":
    file_path = "labeler-vue/public/segments.json"
    remove_long_segments(file_path)
