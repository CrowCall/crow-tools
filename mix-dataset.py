#!/usr/bin/env python3
import os, random, json
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd

random.seed(42)

def background_audio_generator(directory, min_length=6.0, sr=16000):
    files = os.listdir(directory)
    random.shuffle(files)
    for file in files:
        if file.lower().endswith(".mp3"):
            path = os.path.join(directory, file)
            audio, _ = librosa.load(path, sr=sr)
            if len(audio) / sr >= min_length:
                yield audio[:sr*15], file # max length 30 seconds

def get_valid_segments(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    valid = []
    for key, attr in labels.items():
        if attr.get("crowCount") == "single" and not attr.get("badQuality") and not attr.get("human"):
            parts = key.split("-")
            if len(parts) == 3:
                file_id = parts[0]
                start_time = float(parts[1])
                end_time = float(parts[2])
                valid.append({"key": key, "file_id": file_id, "start_time": start_time, "end_time": end_time})
    return valid

def choose_random_segments(valid_segments, count_range=(2,3)):
    count = random.choice(range(count_range[0], count_range[1]+1))
    return random.sample(valid_segments, count)

def adjust_segment(audio, sr, approx_start, approx_end, window_size=0.1, search_range=2.0, threshold=0.01):
    start_idx = int(approx_start * sr)
    end_idx = int(approx_end * sr)
    win_len = int(window_size * sr)
    search_samples = int(search_range * sr)
    quiet_start = None
    for offset in range(0, search_samples, win_len):
        idx = max(0, start_idx - offset)
        if idx + win_len <= len(audio):
            window = audio[idx:idx+win_len]
            if np.mean(np.abs(window)) < threshold:
                quiet_start = idx
                break
    if quiet_start is None:
        return None
    quiet_end = None
    for offset in range(0, search_samples, win_len):
        idx = min(len(audio)-win_len, end_idx + offset)
        window = audio[idx:idx+win_len]
        if np.mean(np.abs(window)) < threshold:
            quiet_end = idx + win_len
            break
    if quiet_end is None:
        return None
    return quiet_start/sr, quiet_end/sr

def generate_random_ir(sr, ir_length_sec):
    ir_length = int(ir_length_sec * sr)
    t = np.linspace(0, ir_length_sec, ir_length)
    # Create a decaying envelope; adjust the decay constant for a stronger tail.
    envelope = np.exp(-t / 0.3)
    # Add some random variation (values between 0.5 and 1.0)
    random_variation = np.random.uniform(0.5, 1.0, size=ir_length)
    ir = envelope * random_variation
    # Ensure the impulse has a strong initial peak.
    ir[0] = 1.0
    return ir

def add_reverb(audio, sr, ir_length_sec=1.0, scale_rirs=10.0):
    reverb_amount = random.uniform(0.0, 0.6)
    # Generate a random impulse response and scale it.
    ir = generate_random_ir(sr, ir_length_sec) * scale_rirs
    # Convolve to get the wet (reverberated) signal.
    wet = np.convolve(audio, ir, mode="full")[:len(audio)]
    # Compute RMS of dry and wet signals.
    dry_rms = np.sqrt(np.mean(audio**2))
    wet_rms = np.sqrt(np.mean(wet**2))
    # Set the wet signal's RMS to, say, 30% of the dry signal's RMS.
    target_wet_rms = 0.3 * dry_rms
    if wet_rms > 0:
        wet = wet * (target_wet_rms / wet_rms)
    # Mix dry and wet signals.
    out = audio + wet
    # Use reverb_amount to interpolate: 0 means fully dry, 1 means fully mixed.
    out = (1 - reverb_amount) * audio + reverb_amount * out
    # Clip to avoid distortion.
    max_val = np.max(np.abs(out))
    if max_val > 1:
        out = out / max_val
    return out

def adjust_volume(audio, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 0.9)
    return audio * factor

def insert_segment(background, segment):
    bg_len = len(background)
    seg_len = len(segment)
    if seg_len >= bg_len:
        return None, None
    start_idx = random.randint(0, bg_len - seg_len)
    layer = np.zeros_like(background)
    layer[start_idx:start_idx+seg_len] = segment
    return start_idx, layer

def mix_audio(background, layers):
    mix = background.copy()
    for layer in layers:
        mix += layer
    max_val = np.max(np.abs(mix))
    if max_val > 1:
        mix = mix / max_val
        layers = [layer / max_val for layer in layers]
    return mix, layers

def main():
    sr = 16000
    preview = False
    backgrounds_dir = "labeler-vue/public/backgrounds"
    labels_json = "labeler-vue/public/labels.json"
    denoised_dir = "labeler-vue/public/library-denoised"
    merged_dir = "labeler-vue/public/mixes/merged"
    separate_dir = "labeler-vue/public/mixes/separate"
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(separate_dir, exist_ok=True)

    bg_generator = background_audio_generator(backgrounds_dir, sr=sr)
    valid_segments = get_valid_segments(labels_json)

    mix_dataset = []
    mix_count = 0

    # Generate 10 mixes (adjust as needed)
    for _ in range(10):
        try:
            background, bg_file = next(bg_generator)
        except StopIteration:
            break

        segments_info = choose_random_segments(valid_segments, (2,3))
        layers = []
        segments_details = []
        for seg in segments_info:
            crow_path = os.path.join(denoised_dir, f"{seg['file_id']}.wav")
            if not os.path.exists(crow_path):
                continue
            crow_audio, _ = librosa.load(crow_path, sr=sr)
            approx_start = seg['start_time']
            approx_end = seg['end_time']
            if int(approx_end * sr) > len(crow_audio):
                continue
            adjusted = adjust_segment(crow_audio, sr, approx_start, approx_end)
            if adjusted is None:
                continue
            adj_start, adj_end = adjusted
            adj_start_sample = int(adj_start * sr)
            adj_end_sample = int(adj_end * sr)
            if adj_end_sample > len(crow_audio):
                continue
            segment_audio = crow_audio[adj_start_sample:adj_end_sample]
            segment_audio = add_reverb(segment_audio, sr)
            segment_audio = adjust_volume(segment_audio)
            start_idx, layer = insert_segment(background, segment_audio)
            if start_idx is None:
                continue
            layers.append(layer)
            segments_details.append({
                "file_id": seg["file_id"],
                "original_key": seg["key"],
                "adjusted_start": adj_start,
                "adjusted_end": adj_end,
                "insertion_index": start_idx
            })
        if not layers:
            continue

        # Mix all audio together
        mix, layers = mix_audio(background, layers)

        if preview:
            print("Previewing background")
            sd.play(background, sr)
            sd.wait()
            for i, layer in enumerate(layers):
                print(f"Previewing segment {i}")
                sd.play(layer, sr)
                sd.wait()
            print("Previewing final mix")
            sd.play(mix, sr)
            sd.wait()
            input("Press Enter to save this mix and continue...")

        mix_filename = f"mix_{mix_count}.wav"
        mix_path = os.path.join(merged_dir, mix_filename)
        sf.write(mix_path, mix, sr)
        # Save individual layers
        layer_files = {}
        bg_filename = f"mix_{mix_count}_background.wav"
        bg_path = os.path.join(separate_dir, bg_filename)
        sf.write(bg_path, background, sr)
        layer_files["background"] = bg_filename
        for i, layer in enumerate(layers):
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

    with open("mix-dataset.json", "w") as f:
        json.dump(mix_dataset, f, indent=2)

if __name__ == "__main__":
    main()
