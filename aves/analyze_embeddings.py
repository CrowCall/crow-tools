import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from sklearn.decomposition import PCA

from ispa import utils
from ispa.features import FeatureBasedISPAPredictor

PATH = os.path.dirname(__file__)
matplotlib.use("TkAgg")

# Initialize the AVES feature predictor.
ispa_f_predictor = FeatureBasedISPAPredictor(
    feature_type='aves',
    kmeans_model=os.path.join(PATH, 'ispa', 'models', 'kmeans.aves.pkl'),
    phoneme_map=os.path.join(PATH, 'ispa', 'models', 'c2p.aves.json'),
    aves_config_path=os.path.join(PATH, 'ispa', 'models', 'aves-base-bio.torchaudio.model_config.json'),
    aves_model_path=os.path.join(PATH, 'ispa', 'models', 'aves-base-bio.torchaudio.pt')
)


def print_label_stats(labels):
    # Convert labels dict into a DataFrame.
    df = pd.DataFrame.from_dict(labels, orient='index')

    # Standardize 'crowCount' values; assume empty means "single".
    df['crowCount'] = df['crowCount'].str.lower().replace('', 'single')

    # Define quality based on badQuality and human flags.
    def determine_quality(row):
        if row.get('human'):
            return 'human'
        elif row.get('badQuality'):
            return 'bad'
        else:
            return 'good'

    df['quality'] = df.apply(determine_quality, axis=1)

    # Overall crosstab: frequency counts.
    overall_ct = pd.crosstab(df['quality'], df['crowCount'], margins=True)

    print("=== LABEL SUMMARY ===")
    print("\nOverall Counts (Quality vs. CrowCount):")
    print(overall_ct)
    print()

    # For each feature, compute frequency (number of True values) and percentages.
    features = ['rattle', 'begging', 'softSong']
    for feat in features:
        # Frequency: since booleans sum to counts.
        feat_freq = df.groupby(['quality', 'crowCount'])[feat].sum().unstack()
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

segments_path = "../labeler-vue/public/segments.json"
segments_dict = json.load(open(segments_path, encoding='utf-8', mode='r'))

labels_path = "../labeler-vue/public/labels.json"
#labels_path = "../labeler-vue/public/auto_labels.json"
labels = json.load(open(labels_path, encoding='utf-8', mode='r'))

print_label_stats(labels)
print_segment_stats(segments_dict)

denoised = False
processed_seconds = 0
sample_rate = 8000
chunk_embeddings = []
chunk_info = []

# Loop through all segments
for file_id, segments in segments_dict.items():
    for segment in segments:
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
        audio_path = os.path.join(f"/home/jonathan/apps/earthspecies/crow-sounds/labeler-vue/public/library{denoised_suffix}/{file_id}.{audio_extension}")

        if os.path.exists(audio_path):
            # check for cached embeddings
            cached_path = os.path.join(f"../embeddings{denoised_suffix}", f"{segment_key}.npy")
            if os.path.exists(cached_path):
                feature = np.load(cached_path)
            else:
                print(f"Skipping {segment_key}, no cached file found")
                continue

            if label:
                if label.get('badQuality') == True or label.get('human') == True:
                    color = "red"
                elif label.get('crowAge') == "juvenile":
                    color = "yellow"
                elif label.get('crowCount') == 'multiple':
                    color = "green"
                # elif label.get('softSong'):
                #     color = "pink"
                elif label.get('rattle'):
                    color = "purple"
                elif label.get('crowCount') == 'single':
                    color = "blue"
                else:
                    color = "gray"
            else:
                color = "gray"

            chunk_embeddings.append(feature)
            chunk_info.append({
                "audio_path": audio_path,
                "chunk_start_time": segment.get('start_time'),
                "chunk_end_time": segment.get('end_time'),
                "color": color
            })
        else:
            print(f"Audio file {audio_path} not found, skipping")

hours = int(processed_seconds // 3600)
minutes = int((processed_seconds % 3600) // 60)
print(f"Processed time: {hours:02d}:{minutes:02d} (HH:MM)")
print(f"Average time per segment: {processed_seconds/len(chunk_embeddings)} seconds")

chunk_embeddings = np.array(chunk_embeddings)
print("Total segments plotted:", chunk_embeddings.shape[0])

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(chunk_embeddings)

# Use custom colors
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

