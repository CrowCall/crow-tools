#!/usr/bin/env python3
import os, random, json
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd

random.seed(42)

# ------------------------------
# Control switches and limits
# ------------------------------
PREVIEW = True
NUM_MIXES = 20000
SAMPLE_RATE = 8000
ENABLE_DENOISED = True
ENABLE_REVERB = False
ENABLE_VOL_NORMALIZATION = True
ENABLE_RANDOM_VOLUME = False
ENABLE_BACKGROUND_SOUNDS = True
RANDOMIZE_BACKGROUND_VOLUME = True
LIMIT_SEGMENT_USE = True
LIMIT_FILE_ID_USE = True
OFFSET_SECONDS = (0.0, 0.5)
MAX_SEGMENT_USES = 1        # Each segment key can be used at most once.
MAX_FILE_ID_USES = 100      # Each file ID can be used at most once.
NUM_SEGMENTS_PER_MIX = 2
BACKGROUND_VOLUME_RANGE = (0.0, 0.9)

# ------------------------------
# Helper functions
# ------------------------------
def background_audio_generator(directory, min_length=3.0, sr=SAMPLE_RATE):
    files = os.listdir(directory)
    while True:
        random.shuffle(files)
        for file in files:
            if file.lower().endswith(".mp3"):
                path = os.path.join(directory, file)
                audio, _ = librosa.load(path, sr=sr)
                if len(audio) / sr >= min_length:
                    yield audio[:sr*8], file

def get_valid_segments(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    valid_segments = []
    for key, attr in labels.items():
        # Only consider valid entries
        if attr.get("crowCount") == 1 and not attr.get("quality") == 1:
            parts = key.split("-")
            if len(parts) == 3:
                file_id = parts[0]
                start_time = float(parts[1])
                end_time = float(parts[2])
                length = end_time - start_time

                # Determine how many splits
                if 15.0 < length < 20.0:
                    n = 3  # each sub-segment ~5s
                elif 8.0 < length < 15.0:
                    n = 2  # each sub-segment up to ~6s
                elif 4.0 < length < 8.0:
                    n = 1  # leave it as-is
                else:
                    n = 1

                # Split the segment into n smaller sub-segments
                sub_segments = []
                if n == 1:
                    sub_segments = [(start_time, end_time)]
                else:
                    sub_len = (end_time - start_time) / n
                    seg_start = start_time
                    for _ in range(n):
                        seg_end = seg_start + sub_len
                        sub_segments.append((seg_start, seg_end))
                        seg_start = seg_end

                # Create valid entries for each sub-segment
                for idx, (sub_start, sub_end) in enumerate(sub_segments):
                    new_key = f"{file_id}-{sub_start:.2f}-{sub_end:.2f}"
                    valid_segments.append({
                        "key": new_key,
                        "file_id": file_id,
                        "start_time": sub_start,
                        "end_time": sub_end
                    })

    return valid_segments

def choose_random_segments(valid_segments, segment_usage, file_id_usage, count=25):
    eligible_segments = []
    for seg in valid_segments:
        key = seg["key"]
        fid = seg["file_id"]
        seg_used = segment_usage.get(key, 0)
        file_used = file_id_usage.get(fid, 0)
        if LIMIT_SEGMENT_USE and seg_used >= MAX_SEGMENT_USES:
            continue
        if LIMIT_FILE_ID_USE and file_used >= MAX_FILE_ID_USES:
            continue
        eligible_segments.append(seg)
    if len(eligible_segments) < count:
        count = len(eligible_segments)
    return random.sample(eligible_segments, count) if eligible_segments else []

def adjust_segment(audio, sr, approx_start, approx_end, window_size=0.1, fraction=0.1):
    """
    Trim leading/trailing silence by computing a smoothed amplitude envelope.
    Returns (start_time_s, end_time_s) in seconds, or None if no active audio is found.
    """
    start_idx = int(approx_start * sr)
    end_idx = int(approx_end * sr)
    segment = audio[start_idx:end_idx]

    # Smooth envelope
    win_len = max(int(window_size * sr), 1)
    envelope = np.convolve(np.abs(segment), np.ones(win_len)/win_len, mode='same')

    # Dynamic threshold: min + fraction*(max-min)
    env_min, env_max = envelope.min(), envelope.max()
    threshold = env_min + fraction * (env_max - env_min)

    # Where does envelope exceed that threshold?
    active_indices = np.where(envelope > threshold)[0]
    if active_indices.size == 0:
        return None

    active_start = active_indices[0]
    active_end   = active_indices[-1] + 1  # +1 to include last active sample

    # Convert back to original audio indices
    new_start_idx = start_idx + active_start
    new_end_idx   = start_idx + active_end

    return new_start_idx / sr, new_end_idx / sr

def generate_random_ir(sr, ir_length_sec):
    ir_length = int(ir_length_sec * sr)
    t = np.linspace(0, ir_length_sec, ir_length)
    envelope = np.exp(-t / 0.3)
    random_variation = np.random.uniform(0.5, 1.0, size=ir_length)
    ir = envelope * random_variation
    ir[0] = 1.0
    return ir

def add_reverb(audio, sr, ir_length_sec=1.0, scale_rirs=10.0):
    reverb_amount = random.uniform(0.0, 0.6)
    ir = generate_random_ir(sr, ir_length_sec) * scale_rirs
    wet = np.convolve(audio, ir, mode="full")[:len(audio)]
    dry_rms = np.sqrt(np.mean(audio**2))
    wet_rms = np.sqrt(np.mean(wet**2))
    target_wet_rms = 0.3 * dry_rms
    if wet_rms > 0:
        wet = wet * (target_wet_rms / wet_rms)
    out = audio + wet
    out = (1 - reverb_amount) * audio + reverb_amount * out
    max_val = np.max(np.abs(out))
    if max_val > 1:
        out = out / max_val
    return out

def is_audio_silient(segment_audio, sr, min_duration=1.0, silence_threshold=1e-1, silence_fraction=0.95):
    # Check if the segment is too short.
    if len(segment_audio) < int(min_duration * sr):
        return True

    # Compute the fraction of silent samples.
    silent_ratio = np.mean(np.abs(segment_audio) < silence_threshold)
    return silent_ratio > silence_fraction

def normalize_audio_segment(audio, target_peak=0.8, target_rms=0.15, max_peak=1.0):
    # Convert integer audio to float in [-1, 1] if needed.
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val

    # Step 1: Peak normalization
    original_peak = np.max(np.abs(audio))
    if original_peak == 0:
        return audio
    scaling_factor = target_peak / original_peak
    normalized_audio = audio * scaling_factor

    # Step 2: Check RMS and apply additional gain if needed.
    rms = np.sqrt(np.mean(normalized_audio ** 2))
    if rms < target_rms:
        # Candidate gain to bring RMS up to target_rms.
        candidate_gain = target_rms / (rms + 1e-8)
        # But after peak normalization, the maximum is target_peak.
        # To prevent clipping above max_peak, we limit the extra gain.
        max_gain = max_peak / target_peak  # maximum multiplier allowed.
        extra_gain = min(candidate_gain, max_gain)
        normalized_audio *= extra_gain

    return normalized_audio

def adjust_volume(audio, factor=None):
    # Original volume adjustment function (unused if normalization is enabled)
    if factor is None:
        factor = random.uniform(0.6, 0.9)
    return audio * factor

def mix_audio(background, segments, sr=SAMPLE_RATE):
    """
    Mix each segment at OFFSET_SECONDS into the background,
    then crop the entire mix to the shortest length among all.
    """
    bg_len = len(background)
    seg_lens = [len(seg) for seg in segments]

    # The final mix length is the minimum among:
    #   - background length
    #   - each segment length
    final_length = min([bg_len] + seg_lens)

    # Truncate the background to final_length
    final_mix = background[:final_length].copy()

    # Create layers for each segment so we can also save them individually
    layers_out = []
    for seg in segments:
        layer = np.zeros(final_length, dtype=seg.dtype)
        seg_len = len(seg)

        # Random offset start
        offset_samples = int(random.uniform(*OFFSET_SECONDS) * sr)

        # Place the segment at OFFSET_SECONDS, but truncate if it goes beyond final_length
        start = offset_samples
        if start >= final_length:
            # No room to place this segment (offset is beyond final length)
            layers_out.append(layer)
            continue

        # If segment extends beyond final_length, crop it
        usable_len = min(seg_len, final_length - offset_samples)
        layer[start:start+usable_len] = seg[:usable_len]

        # Add layer into final mix
        final_mix += layer
        layers_out.append(layer)

    return final_mix, layers_out

# ------------------------------
# Main processing
# ------------------------------
def main():
    sr = SAMPLE_RATE
    total_seconds = 0.0
    mix_dataset_path = ".cache/mixes/mix-dataset.json"
    backgrounds_dir = ".cache/backgrounds"
    labels_json = ".cache/auto_labels.json"
    library_dir = ".cache/library"
    denoised_dir = ".cache/library-denoised"
    merged_dir = ".cache/mixes/merged"
    separate_dir = ".cache/mixes/separate"
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(separate_dir, exist_ok=True)

    # Dictionaries to track usage of segments and file IDs
    segment_usage = {}
    file_id_usage = {}

    if ENABLE_BACKGROUND_SOUNDS:
        bg_generator = background_audio_generator(backgrounds_dir, sr=sr)

    valid_segments = get_valid_segments(labels_json)

    mix_dataset = []
    mix_count = 0

    # Generate mixes (adjust the number as needed)
    for _ in range(NUM_MIXES):
        if ENABLE_BACKGROUND_SOUNDS:
            try:
                background, bg_file = next(bg_generator)
                if RANDOMIZE_BACKGROUND_VOLUME:
                    bg_volume_factor = random.uniform(*BACKGROUND_VOLUME_RANGE)
                    background = background * bg_volume_factor
            except StopIteration:
                break
        else:
            background = np.zeros(sr * 8)  # 8 seconds of silence
            bg_file = "silence"

        # Choose segments while respecting usage limits
        segments_info = choose_random_segments(valid_segments, segment_usage, file_id_usage, count=50)
        segments_list = []  # We'll store each trimmed segment audio here
        segments_details = []

        if not segments_info:
            print(f"No more valid segments found. Skipping...")
            break

        for seg in segments_info:
            if ENABLE_DENOISED:
                crow_path = os.path.join(denoised_dir, f"{seg['file_id']}.wav")
            else:
                crow_path = os.path.join(library_dir, f"{seg['file_id']}.mp3")
            if not os.path.exists(crow_path):
                print("Segment does not exist: {}".format(crow_path))
                continue

            crow_audio, _ = librosa.load(crow_path, sr=sr)
            approx_start = seg['start_time']
            approx_end = seg['end_time']
            if int(approx_end * sr) > len(crow_audio):
                approx_end = len(crow_audio) / sr

            # Normalize segment volume if enabled (else apply a random volume adjustment)
            if ENABLE_VOL_NORMALIZATION:
                crow_audio = normalize_audio_segment(crow_audio, target_peak=0.85)
            elif ENABLE_RANDOM_VOLUME:
                crow_audio = adjust_volume(crow_audio)

            adjusted = adjust_segment(crow_audio, sr, approx_start, approx_end)
            if adjusted is None:
                print("Segment could not be adjusted: {}, start: {}, end: {}".format(crow_path, approx_start, approx_end))
                segment_usage[seg["key"]] = segment_usage.get(seg["key"], 0) + 1
                continue
            adj_start, adj_end = adjusted
            adj_start_sample = int(adj_start * sr)
            adj_end_sample = int(adj_end * sr)
            segment_audio = crow_audio[adj_start_sample:adj_end_sample]

            if ENABLE_VOL_NORMALIZATION:
                segment_audio = normalize_audio_segment(segment_audio, target_peak=0.85)

            # Skip segment if it's less than 1.0 seconds long or if it contains no nonzero values.
            if is_audio_silient(segment_audio, sr):
                print("Segment too short or silent: {}".format(crow_path))
                segment_usage[seg["key"]] = segment_usage.get(seg["key"], 0) + 1
                continue

            # Apply reverb if enabled
            if ENABLE_REVERB:
                segment_audio = add_reverb(segment_audio, sr)

            segments_list.append(segment_audio)
            segments_details.append({
                "file_id": seg["file_id"],
                "original_key": seg["key"],
                "adjusted_start": adj_start,
                "adjusted_end": adj_end
            })

            # Update usage counts
            segment_usage[seg["key"]] = segment_usage.get(seg["key"], 0) + 1
            file_id_usage[seg["file_id"]] = file_id_usage.get(seg["file_id"], 0) + 1

            # For this mix, stop after adding X segments.
            if len(segments_list) == NUM_SEGMENTS_PER_MIX:
                break

        if not segments_list:
            print("No segments inserted")
            continue

        if len(segments_list) < NUM_SEGMENTS_PER_MIX:
            print("Not enough segments inserted: {}".format(len(segments_list)))
            continue

        # Now mix all audio at OFFSET_SECONDS, cropped to the shortest length
        mix, layer_arrays = mix_audio(background, segments_list, sr=sr)

        if PREVIEW:
            print("Previewing background")
            sd.play(background, sr)
            sd.wait()
            for i, seg in enumerate(segments_list):
                print(f"Previewing segment {i}")
                sd.play(seg, sr)
                sd.wait()
            print("Previewing final mix")
            sd.play(mix, sr)
            sd.wait()
            input("Press Enter to save this mix and continue...")

        mix_filename = f"mix_{mix_count}.wav"
        mix_path = os.path.join(merged_dir, mix_filename)
        sf.write(mix_path, mix, sr)

        mix_length = len(mix) / sr
        total_seconds += mix_length
        print(f"Saved {mix_filename}: {mix_count} - {len(layer_arrays)} layers: {mix_length} seconds")

        # Save individual layers.
        layer_files = {}
        bg_filename = f"mix_{mix_count}_background.wav"
        bg_path = os.path.join(separate_dir, bg_filename)
        if ENABLE_BACKGROUND_SOUNDS:
            sf.write(bg_path, background[:len(mix)], sr)
        layer_files["background"] = bg_filename

        for i, layer in enumerate(layer_arrays):
            seg_filename = f"mix_{mix_count}_segment_{i}.wav"
            seg_path = os.path.join(separate_dir, seg_filename)
            sf.write(seg_path, layer, sr)
            layer_files[f"segment_{i}"] = seg_filename

        mix_dataset.append({
            "mix": mix_filename,
            "layers": layer_files,
            "segments": segments_details,
            "background_file": bg_file
        })
        mix_count += 1

    with open(mix_dataset_path, "w") as f:
        json.dump(mix_dataset, f, indent=2)

    print(f"Saved {len(mix_dataset)} mix dataset, Total: {total_seconds/60.0} minutes")

if __name__ == "__main__":
    main()
