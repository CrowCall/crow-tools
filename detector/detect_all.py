#!/usr/bin/env python3
import argparse
import os
import json
import csv
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detector.detect import detect_file_segments
from classifier.classify import predict_embedding
from crowtools.datasets import (
    get_dataset_libraries,
    get_library_dir,
    get_public_libraries,
    get_selected_files,
    read_library_catalog_rows,
    select_catalog_rows,
)

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

def start_detections(libraries=None, selected_ids_by_library=None, cache_base=None):
    libraries = list(libraries) if libraries is not None else get_public_libraries(cache_base)

    for library_name in libraries:
        lib_base = get_library_dir(library_name, cache_base)
        segments_path    = os.path.join(lib_base, "segments.json")
        labels_dir       = os.path.join(lib_base, "labels")
        auto_labels_path = os.path.join(labels_dir, "auto.json")
        embedding_dir    = os.path.join(lib_base, "embeddings")

        os.makedirs(labels_dir, exist_ok=True)

        # Load or initialize auto_labels
        if os.path.exists(auto_labels_path):
            with open(auto_labels_path, "r", encoding="utf-8") as f:
                auto_labels = json.load(f)
        else:
            auto_labels = {}

        # Load or initialize segments
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            segments = {}

        # Determine which files have been processed
        processed_file_ids = {key.split("-")[0] for key in auto_labels.keys()}

        lib_name = os.path.basename(lib_base)
        selected_ids = None if selected_ids_by_library is None else selected_ids_by_library.get(lib_name)
        rows = read_library_catalog_rows(lib_name, cache_base)
        reader = select_catalog_rows(rows, selected_ids=selected_ids)
        for row in reader:
            file_id = row["ML Catalog Number"]
            if file_id in processed_file_ids:
                print(f"[{lib_name}] skip {file_id} (already processed)")
                continue

            print(f"[{lib_name}] detecting {file_id}")
            detections, audio, sr = detect_file_segments(
                file_id,
                public_path=lib_base
            )

            if not detections:
                print(f"[{lib_name}] <<<<<< No detections in file {file_id}")
                continue

            segments[file_id] = []
            embedding_file = os.path.join(embedding_dir, f"{file_id}.npy")
            if not os.path.exists(embedding_file):
                print(f"[{lib_name}] ⚠ Missing embedding for {file_id}, skipping")
                continue
            embeddings_array = np.load(embedding_file)

            for det in detections:
                st = det.get("start_time")
                et = det.get("end_time")
                segment_key = f"{file_id}-{int(st)}-{int(et)}"
                segments[file_id].append({
                    "start_time": st,
                    "end_time": et,
                    "confidence": 0.0,
                    "cluster": 0
                })

                idx = int(st)
                if idx < 0 or idx >= embeddings_array.shape[0]:
                    print(f"[{lib_name}] ⚠ Index {idx} out of range for {segment_key}")
                    continue
                emb = embeddings_array[idx]
                predicted_label = predict_embedding(emb)
                auto_labels[segment_key] = predicted_label

            print(f"[{lib_name}] >>>>> Found {len(detections)} detections in file {file_id}")

        # Save updated segments.json
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        print(f"[{lib_name}] ***** Saved segments for {len(segments)} files to {segments_path}")

        # Save updated auto labels
        with open(auto_labels_path, "w", encoding="utf-8") as f:
            json.dump(auto_labels, f, indent=4)
        print(f"[{lib_name}] ***** Saved auto labels for {len(auto_labels)} segments to {auto_labels_path}\n")

        # Summary statistics
        total_detections = len(auto_labels)
        print(f"[{lib_name}] Total detections: {total_detections}")
        for target in [1, 2, 4]:
            stats = compute_contiguous_stats(segments, auto_labels, target_crowCount=target)
            print(f"[{lib_name}] Contiguous groups (crowCount=={target}): {stats}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect crow segments across a dataset.")
    parser.add_argument("--dataset", default=None, help="Dataset to process. Defaults to all discovered public libraries.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    args = parser.parse_args(argv)

    if args.dataset:
        libraries = get_dataset_libraries(args.dataset, args.cache_dir)
        selected_ids_by_library = get_selected_files(args.dataset, args.cache_dir)
    else:
        libraries = None
        selected_ids_by_library = None

    start_detections(
        libraries=libraries,
        selected_ids_by_library=selected_ids_by_library,
        cache_base=args.cache_dir,
    )


if __name__ == "__main__":
    main()
