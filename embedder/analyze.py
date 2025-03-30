import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from sklearn.decomposition import PCA
from ispa import utils

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
        if row.get('quality') == 1:
            return 'bad'
        else:
            return 'good'

    df['quality_label'] = df.apply(determine_quality, axis=1)

    # Overall crosstab: frequency counts.
    overall_ct = pd.crosstab(df['quality_label'], df['crowCount'], margins=True)

    print("=== LABEL SUMMARY ===")
    print("\nOverall Counts (Quality vs. CrowCount):")
    print(overall_ct)
    print()

    # For each feature, compute frequency (number of True values) and percentages.
    features = ['rattle', 'begging', 'softSong']
    for feat in features:
        # Frequency: since booleans sum to counts.
        feat_freq = df.groupby(['quality_label', 'crowCount'])[feat].sum().unstack()
        print(f"{feat.capitalize()} Counts (True):")
        print(feat_freq.fillna(0).astype(int))
        print()

    print("Total labels:", len(df))
    print()

def print_segment_stats(segments_dict):
    total_files = len(segments_dict)
    segments_list = []

    for segs in segments_dict.values():
        # Ensure we have a list.
        seg_list = segs if isinstance(segs, list) else [segs]
        for seg in seg_list:
            duration = seg["end_time"] - seg["start_time"]
            segments_list.append(duration)

    total_segments = len(segments_list)
    durations_series = pd.Series(segments_list)
    freq = durations_series.value_counts().sort_index()
    pct = durations_series.value_counts(normalize=True).sort_index() * 100
    freq_table = pd.DataFrame({'Count': freq, 'Percentage': pct.round(1)})

    print("=== SEGMENT SUMMARY ===")
    print(f"Total files   : {total_files}")
    print(f"Total segments: {total_segments}")
    print("\nSegment Duration Distribution (seconds):")
    print(freq_table)
    print()

segments_path = os.path.join("..", ".cache", "cluster_segments.json")
segments_dict = json.load(open(segments_path, encoding='utf-8', mode='r'))

labels_path = os.path.join("..", ".cache", "cluster_labels.json")
labels = json.load(open(labels_path, encoding='utf-8', mode='r'))

print_label_stats(labels)
print_segment_stats(segments_dict)

denoised = True
processed_seconds = 0
sample_rate = 8000
chunk_embeddings = []
chunk_info = []

# Loop through all segments
for file_id, segments in segments_dict.items():
    for segment in segments:
        # Compute segment key using file_id only (as in "225318321-0-3")
        segment_key = f"{file_id}-{segment.get('start_time'):.0f}-{segment.get('end_time'):.0f}"
        label = labels.get(segment_key)

        # Get labeled segment (if any)
        # if not (label and label.get('crowCount') == 'single'
        #         and label.get('badQuality') == False
        #         and label.get('human') == False):
        #     continue
        if not label:
            continue

        # Get segment length
        segment_length = segment.get('end_time') - segment.get('start_time')
        processed_seconds += segment_length

        # Get file audio file
        denoised_suffix = ""
        audio_extension = "mp3"
        if denoised:
            denoised_suffix = "-denoised"
            audio_extension = "wav"
        audio_path = os.path.join(PATH, "..", ".cache", f"library{denoised_suffix}", f"{file_id}.{audio_extension}")

        if os.path.exists(audio_path):
            # check for cached embeddings
            cached_path = os.path.join(PATH, "..", ".cache", f"embeddings{denoised_suffix}", f"{file_id}.npy")
            if os.path.exists(cached_path):
                feature = np.load(cached_path)
            else:
                print(f"Skipping {segment_key}, no cached file found")
                continue

            if label:
                if label.get('quality') == 1:
                    color = "red"
                elif label.get('crowAge') == 2:
                    color = "yellow"
                elif label.get('crowCount') == 2:
                    color = "green"
                # elif label.get('softSong'):
                #     color = "pink"
                elif label.get('rattle'):
                    color = "purple"
                elif label.get('crowCount') == 1:
                    color = "blue"
                else:
                    color = "gray"
            else:
                color = "gray"

            chunk_embeddings.append(feature[int(segment.get('start_time')):int(segment.get('end_time'))])
            chunk_info.append({
                "audio_path": audio_path,
                "chunk_start_time": segment.get('start_time'),
                "chunk_end_time": segment.get('end_time'),
                "color": color,
                "segment_key": segment_key
            })
        else:
            print(f"Audio file {audio_path} not found, skipping")

hours = int(processed_seconds // 3600)
minutes = int((processed_seconds % 3600) // 60)
print(f"Processed time: {hours:02d}:{minutes:02d} (HH:MM)")
print(f"Average time per segment: {processed_seconds/len(chunk_embeddings)} seconds")

chunk_embeddings = np.array(chunk_embeddings)
print("Total segments plotted:", chunk_embeddings.shape[0])

# Perform PCA: reduce from 768 to 3 dimensions.
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(chunk_embeddings.squeeze(1))

# Build output data: for each embedding, store segment_key, 3D coordinates, and color.
output_data = []
embeddings_3d_list = embeddings_3d.tolist()
for i, coords in enumerate(embeddings_3d_list):
    seg_key = chunk_info[i].get("segment_key")
    color = chunk_info[i].get("color")
    output_data.append({
         "segment_key": seg_key,
         "coordinates": coords,
         "color": color
    })

# Save the output data to JSON.
output_json_path = os.path.join(PATH, "..", ".cache", "embeddings-3d.json")
with open(output_json_path, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Saved 3D embeddings to {output_json_path}")

# Proceed with plotting.
colors = [info.get('color') for info in chunk_info]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set picker=True so points can be clicked
sc = ax.scatter(embeddings_3d[:, 0],
                embeddings_3d[:, 1],
                embeddings_3d[:, 2],
                s=10, c=colors, picker=5)

ax.set_title("3D PCA of AVES Embedding Chunks (Interactive)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")

def on_pick(event):
    # event.ind is an array of the point indices that were clicked
    ind = event.ind
    if len(ind) == 0:
        return

    idx = ind[0]  # Just take the first index if multiple
    info = chunk_info[idx]

    audio_path = info["audio_path"]
    start_t = info["chunk_start_time"]
    end_t = info["chunk_end_time"]

    # Print info to console
    print(f"\nClicked chunk #{idx}")
    print(f"File: {audio_path}")
    print(f"Time: {start_t:.2f}s to {end_t:.2f}s")

    # Reload just this chunk for playback
    chunk_waveform, _ = utils.load_waveform(audio_path, tgt_sr=sample_rate, start_sec=start_t, end_sec=end_t)

    # Convert to a NumPy array if needed for sounddevice
    chunk_np = chunk_waveform.squeeze(0).detach().numpy()
    sd.play(chunk_np, sample_rate)  # Playback using sounddevice

# Connect the pick event to the callback
fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()

