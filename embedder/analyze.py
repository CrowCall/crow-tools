import json
import os
import random
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from sklearn.decomposition import PCA
from matplotlib.widgets import RadioButtons
from ispa import utils
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowtools.datasets import (
    get_dataset_artifact_path,
    load_dataset_auto_labels,
    load_dataset_segments,
    resolve_dataset_audio_path,
    resolve_dataset_embedding_path,
)

PATH = os.path.dirname(__file__)

def is_headless():
    return not os.environ.get("DISPLAY") and os.name != "nt"
if is_headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")


def print_label_stats(labels):
    # Convert labels dict into a DataFrame.
    df = pd.DataFrame.from_dict(labels, orient='index')

    # Define quality based on badQuality and human flags.
    def determine_quality(row):
        return 'bad' if row.get('quality') == 1 else 'good'

    df['quality_label'] = df.apply(determine_quality, axis=1)

    # Overall crosstab: frequency counts.
    overall_ct = pd.crosstab(df['quality_label'], df['crowCount'], margins=True)

    print("=== LABEL SUMMARY ===")
    print("\nOverall Counts (Quality vs. CrowCount):")
    print(overall_ct)
    print()

    # For each feature, compute frequency (number of True values) and percentages.
    for feat in ['rattle', 'begging', 'softSong', 'mob', 'alert']:
        feat_freq = df.groupby(['quality_label', 'crowCount'])[feat].sum().unstack()
        print(f"{feat.capitalize()} Counts (True):")
        print(feat_freq.fillna(0).astype(int))
        print()

    print("Total labels:", len(df))
    print()

def print_segment_stats(segments_dict):
    segments_list = []
    for segs in segments_dict.values():
        seg_list = segs if isinstance(segs, list) else [segs]
        for seg in seg_list:
            duration = seg["end_time"] - seg["start_time"]
            segments_list.append(duration)
    durations_series = pd.Series(segments_list)
    freq = durations_series.value_counts().sort_index()
    pct = durations_series.value_counts(normalize=True).sort_index() * 100
    freq_table = pd.DataFrame({'Count': freq, 'Percentage': pct.round(1)})

    print("=== SEGMENT SUMMARY ===")
    print(f"Total files   : {len(segments_dict)}")
    print(f"Total segments: {len(segments_list)}")
    print("\nSegment Duration Distribution (seconds):")
    print(freq_table)
    print()

def choose_color(label, mode='quality'):
    # For non-quality modes, hide bad quality points (make transparent)
    if mode != 'quality' and label.get('quality') == 1:
        return (0, 0, 0, 0)  # Invisible

    if mode == 'quality':
        if label.get('quality') == 1:
            return "red"
        else:
            return "limegreen"  # bright, but not overly neon

    elif mode == 'crowAge':
        age = label.get('crowAge')
        if age == 1:
            return "dodgerblue"  # a friendly, bright blue
        elif age == 2:
            return "gold"        # a warm, appealing yellow
        else:
            return "gray"

    elif mode == 'crowCount':
        count = label.get('crowCount', 0)
        if count == 0:
            return "lightgray"
        elif count == 1:
            return "#00bfff"
        elif count == 2:
            return "orange"
        elif count == 4:
            return "orchid"          # soft purple
        else:
            return "gray"

    elif mode == 'features':
        # Priority order for binary features: alert > mob > begging > softSong > rattle.
        features_priority = ['alert', 'mob', 'begging', 'softSong', 'rattle']
        features_colors = {
            'alert': "dodgerblue",       # bright blue
            'mob': "mediumorchid",       # appealing purple
            'begging': "goldenrod",      # warm golden tone
            'softSong': "mediumseagreen",# calm green
            'rattle': "tomato"           # pleasant red-orange
        }
        for feat in features_priority:
            if label.get(feat):
                return features_colors[feat]
        return "gray"

    return "gray"


def main(sample_size, dataset_name="all-public", cache_base=None, show_plot=True):
    segments_dict = load_dataset_segments(dataset_name, cache_base=cache_base)
    labels = load_dataset_auto_labels(dataset_name, cache_base=cache_base)

    print_label_stats(labels)
    print_segment_stats(segments_dict)

    denoised = True
    processed_seconds = 0

    valid_segments = []
    for file_id, segs in segments_dict.items():
        for segment in segs:
            segment_key = f"{file_id}-{segment.get('start_time'):.0f}-{segment.get('end_time'):.0f}"
            if segment_key not in labels:
                continue

            valid_segments.append({
                "file_id": file_id,
                "segment": segment,
                "segment_key": segment_key,
                "label": labels[segment_key]
            })

    if sample_size < len(valid_segments):
        valid_segments = random.sample(valid_segments, sample_size)
        print(f"Randomly sampled {sample_size} segments from available data.")
    else:
        print(f"Processing all {len(valid_segments)} available segments.")

    chunk_embeddings = []
    chunk_info = []  # Each item holds audio path, times, initial color, segment_key, and label.
    sample_rate = 8000

    for entry in valid_segments:
        file_id = entry["file_id"]
        segment = entry["segment"]
        segment_key = entry["segment_key"]
        label = entry["label"]

        segment_length = segment.get('end_time') - segment.get('start_time')
        processed_seconds += segment_length

        denoised_suffix = "-denoised" if denoised else ""
        audio_extension = "wav" if denoised else "mp3"
        audio_path = resolve_dataset_audio_path(
            dataset_name,
            file_id,
            denoised=denoised,
            cache_base=cache_base,
        )

        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found, skipping")
            continue

        cached_path = resolve_dataset_embedding_path(
            dataset_name,
            file_id,
            denoised=denoised,
            cache_base=cache_base,
        )
        if not os.path.exists(cached_path):
            print(f"Skipping {segment_key}, no cached embedding file found")
            continue

        try:
            feature = np.load(cached_path, mmap_mode='r')
        except Exception as e:
            print(f"Error loading {cached_path}: {e}")
            continue

        # Set initial color using 'quality' mode.
        color = choose_color(label, mode='quality')

        start_idx = int(segment.get('start_time'))
        end_idx = int(segment.get('end_time'))
        chunk = feature[start_idx:end_idx]
        chunk_embeddings.append(chunk)
        chunk_info.append({
            "audio_path": audio_path,
            "chunk_start_time": segment.get('start_time'),
            "chunk_end_time": segment.get('end_time'),
            "color": color,
            "segment_key": segment_key,
            "label": label
        })

    if len(chunk_embeddings) == 0:
        print("No segments to process after filtering. Exiting.")
        return

    hours = int(processed_seconds // 3600)
    minutes = int((processed_seconds % 3600) // 60)
    print(f"Processed time: {hours:02d}:{minutes:02d} (HH:MM)")
    print(f"Average time per segment: {processed_seconds / len(chunk_embeddings):.2f} seconds")

    chunk_embeddings = np.array(chunk_embeddings)
    print("Total segments for PCA:", chunk_embeddings.shape[0])
    if len(chunk_embeddings.shape) != 2:
        chunk_embeddings = chunk_embeddings.squeeze(1)

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(chunk_embeddings)

    output_json_path = get_dataset_artifact_path(dataset_name, "embeddings-3d.json", cache_base=cache_base)
    output_data = []
    for i, coords in enumerate(embeddings_3d.tolist()):
        output_data.append({
            "segment_key": chunk_info[i]["segment_key"],
            "coordinates": coords,
            "color": chunk_info[i]["color"]
        })

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved 3D embeddings to {output_json_path}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    base_marker_size = 10
    colors = [info["color"] for info in chunk_info]
    sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                    s=base_marker_size, c=colors, picker=True)

    ax.set_title("3D PCA of AVES Embedding Chunks (Interactive)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    def update_output_json(new_colors):
        updated_data = []
        for i, coords in enumerate(embeddings_3d.tolist()):
            updated_data.append({
                "segment_key": chunk_info[i]["segment_key"],
                "coordinates": coords,
                "color": new_colors[i]
            })
        with open(output_json_path, "w") as f:
            json.dump(updated_data, f, indent=2)
        print(f"Re-saved updated output JSON to {output_json_path}")

    def update_mode(mode):
        new_colors = [choose_color(info["label"], mode=mode) for info in chunk_info]
        sc.set_color(new_colors)
        plt.draw()
        update_output_json(new_colors)

    ax_mode = plt.axes([0.05, 0.8, 0.12, 0.08])
    radio = RadioButtons(ax_mode, ('quality', 'crowAge', 'crowCount', 'features'), active=0)
    radio.on_clicked(update_mode)

    def on_scroll(event):
        if event.button == 'up':
            ax.dist = max(ax.dist / 1.1, 1)
        elif event.button == 'down':
            ax.dist = ax.dist * 1.1

        factor = 1.2 if event.button == 'up' else 1 / 1.2 if event.button == 'down' else 1
        current_sizes = sc.get_sizes()
        new_sizes = current_sizes * factor
        sc.set_sizes(new_sizes)
        plt.draw()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_pick(event):
        ind = event.ind
        if len(ind) == 0:
            return
        idx = ind[0]
        info = chunk_info[idx]
        print(f"\nClicked segment #{idx}")
        print(f"File: {info['audio_path']}")
        print(f"Time: {info['chunk_start_time']:.2f}s to {info['chunk_end_time']:.2f}s")
        chunk_waveform, _ = utils.load_waveform(info["audio_path"],
                                                tgt_sr=sample_rate,
                                                start_sec=info["chunk_start_time"],
                                                end_sec=info["chunk_end_time"])
        chunk_np = chunk_waveform.squeeze(0).detach().numpy()
        sd.play(chunk_np, sample_rate)

    fig.canvas.mpl_connect('pick_event', on_pick)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Embedding Detections with Interactive Color Modes")
    parser.add_argument("--dataset", default="all-public", help="Dataset to analyze.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Number of segments to sample (default: 10000)")
    parser.add_argument("--no-show", action="store_true", help="Generate outputs without opening the interactive plot.")
    args = parser.parse_args()
    main(args.sample_size, dataset_name=args.dataset, cache_base=args.cache_dir, show_plot=not args.no_show)
