import json
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sounddevice as sd
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

segments_path = "../labeler-vue/public/segments.json"
segments_dict = json.load(open(segments_path, encoding='utf-8', mode='r'))

labels_path = "../labeler-vue/public/labels.json"
labels = json.load(open(labels_path, encoding='utf-8', mode='r'))

# Print stats from labels
print(f"Total labels: {len(labels)}")
print("----Sources:")
print(f"single crows (total): {len([value for key, value in labels.items() if value.get('crowCount') == 'single'])}")
print(f"single crows (good): {len([value for key, value in labels.items() if value.get('crowCount') == 'single' and value.get('badQuality') == False and value.get('human') == False])}")
print(f"multiple crows (total): {len([value for key, value in labels.items() if value.get('crowCount') == 'multiple'])}")
print("----Age:")
print(f"adult crows: {len([value for key, value in labels.items() if value.get('crowAge') == 'adult'])}")
print(f"juvenile crows: {len([value for key, value in labels.items() if value.get('crowAge') == 'juvenile'])}")
print("----Features:")
print(f"rattle crows: {len([value for key, value in labels.items() if value.get('rattle') == True])}")
print(f"begging crows: {len([value for key, value in labels.items() if value.get('begging') == True])}")
print(f"soft-song crows: {len([value for key, value in labels.items() if value.get('softSong') == True])}")
print("----Quality:")
print(f"good quality: {len([value for key, value in labels.items() if value.get('badQuality') == False and value.get('human') == False])}")
print(f"bad quality: {len([value for key, value in labels.items() if value.get('badQuality') == True or value.get('human') == True])}")
print("----")

denoised = False
processed_seconds = 0
time_resolution = 50.0
sample_rate = 16000
chunk_size = 149
chunk_embeddings = []
chunk_info = []

# Loop through all segments
for file_id, segments in segments_dict.items():
    for segment in segments:
        segment_key = f"{file_id}-{segment.get('start_time'):.0f}-{segment.get('end_time'):.0f}"
        label = labels.get(segment_key)

        if not label:
            continue
        # Get labeled segment (if any)
        # if not (label and label.get('crowCount') == 'single'
        #         and label.get('badQuality') == False
        #         and label.get('human') == False):
        #     continue

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
            cached_path = os.path.join(f"embeddings{denoised_suffix}", f"{segment_key}.npy")
            if os.path.exists(cached_path):
                feature = np.load(cached_path)
            else:
                # Load the segment of audio
                waveform, _ = utils.load_waveform(audio_path, start_sec=segment.get('start_time'), end_sec=segment.get('end_time'))

                feature_chunk_length = int(sample_rate * 60)
                if waveform.shape[-1] > feature_chunk_length:
                    features = []
                    # Process each chunk
                    for start in range(0, waveform.shape[-1], feature_chunk_length):
                        end = min(start + feature_chunk_length, waveform.shape[-1])
                        waveform_chunk = waveform[..., start:end]

                        # Extract features using AVES on the chunk
                        feature_chunk = ispa_f_predictor.feature_extractor(waveform_chunk)  # (batch, time, feature)
                        feature_chunk = feature_chunk.squeeze(0)  # (time, feature)
                        feature_chunk = feature_chunk.detach().numpy()  # Convert to NumPy
                        features.append(feature_chunk)

                    # Concatenate features along the time axis
                    feature = np.concatenate(features, axis=0)
                else:
                    # Process the whole segment if it is 60 seconds or shorter
                    feature = ispa_f_predictor.feature_extractor(waveform)  # (batch, time, feature)
                    feature = feature.squeeze(0)  # (time, feature)
                    feature = feature.detach().numpy()  # Convert to NumPy

                # Save to cache
                np.save(cached_path, feature)


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

            num_chunks = feature.shape[0] // chunk_size
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                chunk_data = feature[start_idx:end_idx, :]  # (chunk_size, feature_dim)
                chunk_mean = np.mean(chunk_data, axis=0)    # (feature_dim,)

                # Store the embedding
                chunk_embeddings.append(chunk_mean)

                # Store metadata for playback
                chunk_start_time = segment.get('start_time') + (start_idx / time_resolution)
                chunk_end_time = segment.get('start_time') + (end_idx / time_resolution)

                chunk_info.append({
                    "audio_path": audio_path,
                    "chunk_start_time": chunk_start_time,
                    "chunk_end_time": chunk_end_time,
                    "color": color
                })

            # Handle leftover frames
            leftover = feature.shape[0] % chunk_size
            if leftover > 49:
                start_idx = feature.shape[0] - leftover
                chunk_data = feature[start_idx:, :]
                chunk_mean = np.mean(chunk_data, axis=0)

                chunk_embeddings.append(chunk_mean)
                chunk_start_time = segment.get('start_time') + (start_idx / time_resolution)
                chunk_end_time = segment.get('start_time') + ((start_idx + leftover) / time_resolution)

                chunk_info.append({
                    "audio_path": audio_path,
                    "chunk_start_time": chunk_start_time,
                    "chunk_end_time": chunk_end_time,
                    "color": color
                })
        else:
            print(f"Audio file {audio_path} not found, skipping")

print(f"Processed minutes: {processed_seconds / 60} minutes")


chunk_embeddings = np.array(chunk_embeddings)
print("Total chunks:", chunk_embeddings.shape[0])

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