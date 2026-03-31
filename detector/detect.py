#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
import shutil
from datetime import datetime
import tempfile

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
from tqdm import tqdm

# Import tkinter components for UI dialogs.
import tkinter as tk
from tkinter import messagebox, simpledialog
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classifier.classify import predict_embedding
from embedder.embed import generate_embeddings
from crowtools.datasets import (
    ensure_dataset,
    get_dataset_artifact_path,
    get_dataset_import_path,
    get_dataset_libraries,
    resolve_dataset_file_library,
)

def is_headless():
    return not os.environ.get("DISPLAY") and os.name != "nt"
if is_headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

###############################################################################
# Helper function to retrieve embeddings, volumes, audio and sample rate.
###############################################################################
def get_data(arg, public_path=None, include_audio=False):
    """
    Given a file identifier or full file path, returns:
      - embeddings: a 2D numpy array (num_seconds x feature_dim)
      - volumes: a 1D numpy array of per-second volume metrics
      - audio: the audio waveform (mono)
      - sr: sample rate

    If arg is a valid file path, the function will:
      - Load the audio (forcing a sample rate of 8000 Hz for consistency)
      - Generate raw embeddings via generate_embeddings()
      - Average every 25 frames (i.e. ~1 second) into one embedding vector
      - Compute per‑second volume metrics from the waveform
      (No files are written to disk.)

    Otherwise, it assumes arg is a file ID and attempts to load cached files from:
      - {public_path}/embeddings-denoised/{file_id}.npy
      - {public_path}/embeddings-denoised-volumes/{file_id}.npy
      - {public_path}/audio/{file_id}.mp3
    """
    if public_path is None and not os.path.isfile(arg):
        raise ValueError("public_path is required when resolving a cached file ID")

    if os.path.isfile(arg):
        # Uncached mode: arg is a file path.
        print(f"Processing uncached file: {arg}")
        sample_rate = 8000  # force consistent sample rate (as in embed_all.py)
        cache_path = os.path.join(tempfile.gettempdir(), os.path.basename(arg) + ".avg_embeddings.npy")
        try:
            audio, sr = librosa.load(arg, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio from {arg}: {e}")
            sys.exit(1)
        if os.path.exists(cache_path):
            embeddings = np.load(cache_path)
        else:
            print("Generating embeddings ...")
            full_embeddings = generate_embeddings(audio)
            full_embeddings = np.array(full_embeddings, dtype=np.float32)
            num_frames = full_embeddings.shape[0]
            # Average every 25 frames to get one embedding per second.
            chunk_size = 25
            num_chunks = int(np.ceil(num_frames / chunk_size))
            embedding_means = []
            for i in tqdm(range(num_chunks)):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_frames)
                chunk = full_embeddings[start_idx:end_idx]
                mean_emb = np.mean(chunk, axis=0)
                embedding_means.append(mean_emb)
            embeddings = np.stack(embedding_means, axis=0)
            np.save(cache_path, embeddings)
        # Compute per-second volume metrics (mean absolute amplitude).
        total_samples = len(audio)
        total_seconds = int(np.ceil(total_samples / sr))
        volumes = []
        for sec in tqdm(range(total_seconds)):
            start_samp = sec * sr
            end_samp = min((sec + 1) * sr, total_samples)
            segment = audio[start_samp:end_samp]
            mean_amp = np.mean(np.abs(segment)) if len(segment) > 0 else 0.0
            volumes.append(mean_amp)
        volumes = np.array(volumes, dtype=np.float32)
        return embeddings, volumes, audio, sr

    else:
        # Cached mode: arg is a file ID.
        file_id = arg
        embeddings_path = os.path.join(public_path, "embeddings", f"{file_id}.npy")
        volumes_path = os.path.join(public_path, "embeddings-denoised-volumes", f"{file_id}.npy")
        library_dir = os.path.join(public_path, "audio")
        audio_path = os.path.join(library_dir, f"{file_id}.mp3")

        if not os.path.exists(embeddings_path):
            print(f"Embedding file not found for file ID {file_id}")
            sys.exit(1)
        if not os.path.exists(volumes_path):
            print(f"Volume file not found for file ID {file_id}")
            sys.exit(1)
        if not os.path.exists(audio_path):
            print(f"Audio file not found for file ID {file_id}")
            sys.exit(1)

        embeddings = np.load(embeddings_path)
        volumes = np.load(volumes_path)
        if volumes.ndim > 1:
            volumes = volumes.squeeze(-1)
        try:
            if include_audio:
                audio, sr = librosa.load(audio_path, sr=8000, mono=True)
            else:
                audio = None
                sr = 8000
        except Exception as e:
            print(f"Error loading audio: {e}")
            sys.exit(1)
        return embeddings, volumes, audio, sr

###############################################################################
# Detection function (common for both cached and uncached data)
###############################################################################
def detect_file_segments(arg, volume_threshold=0.0002, device=None, public_path=None, include_audio=False):
    """
    Loads embeddings and volume data (and audio) using get_data() and then runs
    the classifier on each second where the volume exceeds the threshold.
    Returns:
      - detections: list of detection dictionaries
      - audio: the audio waveform
      - sr: sample rate
    """
    embeddings, volumes, audio, sr = get_data(arg, public_path, include_audio)
    detections = []
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Process seconds based on the lesser of embeddings length and volume length.
    num_seconds = min(embeddings.shape[0], len(volumes))
    for i in tqdm(range(num_seconds)):
        if volumes[i] <= volume_threshold:
            continue
        emb = embeddings[i]
        pred = predict_embedding(emb, device=device)
        # Only consider valid detections (quality > 1 and crowCount > 0).
        if pred["quality"] > 1 and pred["crowCount"] > 0:
            detection = {
                "start_time": float(i),
                "end_time": float(i + 1),
                "crowCount": pred["crowCount"],
                "crowAge": pred["crowAge"],
                "alert": pred["alert"],
                "begging": pred["begging"],
                "softSong": pred["softSong"],
                "rattle": pred["rattle"],
                "mob": pred["mob"],
                "quality": pred["quality"],
            }
            detections.append(detection)
    return detections, audio, sr

###############################################################################
# Generate detection summary (unchanged)
###############################################################################
def generate_detection_summary(detections, gap=5):
    """
    Returns a natural language paragraph summarizing high-quality detections.
    Only detections with quality==2 and at least one binary attribute True are considered.
    For each attribute (e.g., "softSong", "rattle", etc.) and for each natural description (crow count and age),
    only the first start time of each contiguous block (gaps > gap seconds) is listed.

    The start times are formatted as mm:ss.
    """
    # Define binary detection attributes.
    binary_keys = ["alert", "begging", "softSong", "rattle", "mob"]

    # Filter detections.
    filtered = [d for d in detections if d.get("quality") == 2 and any(d.get(attr, False) for attr in binary_keys)]
    if not filtered:
        return "No crow detections were found in the audio file."

    # Helper: Format seconds as mm:ss.
    def format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    # Helper: Collapse contiguous timestamps (sorted ascending) if difference <= gap seconds.
    def collapse_timestamps(times, gap=5):
        if not times:
            return []
        times = sorted(times)
        collapsed = [times[0]]
        last = times[0]
        for t in times[1:]:
            if t - last > gap:
                collapsed.append(t)
                last = t
        return collapsed

    # Helper: Natural language description for crow count.
    def crow_count_desc(count):
        if count == 1:
            return "a single crow"
        elif count == 2:
            return "a pair of crows"
        else:
            return f"a group of crows"

    # Helper: Natural language description for crow age.
    def crow_age_desc(age):
        return "adult" if age == 1 else "juvenile" if age == 2 else "of unknown age"

    # Group detections by binary attribute and natural description.
    groups = {}  # key: (attribute, description), value: list of start times (floats)
    for det in filtered:
        start = det.get("start_time")
        if start is None:
            continue
        count = det.get("crowCount", 1)
        age = det.get("crowAge", 1)
        description = f"{crow_count_desc(count)}, {crow_age_desc(age)}"
        for attr in binary_keys:
            if det.get(attr, False):
                key = (attr, description)
                groups.setdefault(key, []).append(float(start))

    # Build the paragraph summary.
    summary_parts = ["The following detections were observed in this recording:"]
    for (attr, description), times in groups.items():
        collapsed = collapse_timestamps(times, gap)
        if not collapsed:
            continue
        attr_nl = attr.replace("softSong", "subsong")
        times_str = ", ".join(format_time(t) for t in collapsed)
        num_groups = len(collapsed)
        summary_parts.append(f"{num_groups} {attr_nl} calls ({description}) at {times_str}.")
    return "\n".join(summary_parts)

###############################################################################
# INTERACTIVE TIMELINE PLAYER
###############################################################################
class TimelinePlayer:
    def __init__(self, label, detections, audio, sr, public_path=None, raw_file=None, dataset_name="all-public", cache_base=None):
        """
        Args:
          label (str): file identifier or base filename.
          detections (list): list of detection dictionaries.
          audio (np.ndarray): audio waveform (mono).
          sr (int): sample rate.
          public_path (str): base path for .cache directory.
          raw_file (str): original file path if uncached raw file was provided;
                          if None, then a cached file ID was provided and the
                          "Add to Labeler" functionality is disabled.
        """
        self.label = label
        self.detections = detections
        self.audio = audio
        self.sr = sr
        self.total_duration = len(audio) / sr
        self.playing = False
        self.play_offset = 0.0  # seconds; playback start position
        self.play_start_time = None

        if public_path is None:
            self.public_path = os.path.join(os.path.dirname(__file__), "..", ".cache")
        else:
            self.public_path = public_path

        # Store the original raw file (if provided).
        self.source_file = raw_file
        self.dataset_name = dataset_name
        self.cache_base = cache_base

        # Define the boolean features to display.
        self.features = ["alert", "mob", "begging", "softSong", "rattle", "other"]
        self.feature_intervals = self.compute_feature_intervals()

        # Create a figure with two subplots: waveform and detection timeline.
        self.fig, (self.ax_wave, self.ax_det) = plt.subplots(2, 1, sharex=True, figsize=(19.2, 8), dpi=100)
        self.fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)
        self.fig.suptitle(f"File: {label} - Detection Timeline", fontsize=16)

        # Downsample waveform for efficiency.
        target_points = 1920
        if len(self.audio) > target_points:
            indices = np.linspace(0, len(self.audio) - 1, target_points).astype(int)
            downsampled_audio = self.audio[indices]
            t = np.linspace(0, self.total_duration, target_points)
        else:
            downsampled_audio = self.audio
            t = np.linspace(0, self.total_duration, len(self.audio))
        self.ax_wave.plot(t, downsampled_audio, color="blue", lw=0.8)
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_xlim(0, self.total_duration)
        self.ax_wave.set_title("Waveform")

        # Plot detection timeline.
        self.ax_det.set_title("Detections")
        self.ax_det.set_ylim(0, len(self.features))
        self.ax_det.set_yticks([i + 0.5 for i in range(len(self.features))])
        self.ax_det.set_yticklabels(self.features)
        self.ax_det.set_xlabel("Time (s)")
        self.ax_det.set_xticks(np.linspace(0, self.total_duration, num=5))
        self.ax_det.set_xlim(0, self.total_duration)
        self.ax_det.set_autoscale_on(False)

        # Draw detection bars.
        colors = {"alert": "red", "mob": "green", "begging": "orange",
                  "softSong": "purple", "rattle": "brown", "other": "gray"}
        for i, feat in enumerate(self.features):
            intervals = self.feature_intervals.get(feat, [])
            for (start, end) in intervals:
                self.ax_det.hlines(y=i + 0.5, xmin=start, xmax=end,
                                   colors=colors.get(feat, "gray"), linewidth=8)

        # Add a vertical playhead.
        self.playhead_line = self.ax_det.axvline(x=0, color="black", linestyle="--", lw=2)
        self.playhead_line.set_visible(False)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax_det.bbox)
        self.playhead_line.set_visible(True)

        # Add a Play/Pause button.
        self.button_ax = self.fig.add_axes([0.87, 0.05, 0.1, 0.05])
        self.play_button = plt.Button(self.button_ax, "Play")
        self.play_button.on_clicked(self.toggle_play)

        # Only add the "Add to Labeler" button if a raw file path was provided.
        if self.source_file is not None:
            self.add_button_ax = self.fig.add_axes([0.87, 0.12, 0.1, 0.05])
            self.add_button = plt.Button(self.add_button_ax, "Add to Labeler")
            self.add_button.on_clicked(self.add_to_labeler)

        # Connect mouse click and draw events.
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)

        # Set up a timer to update the playhead.
        self.timer = self.fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.update_playhead)
        self.timer.start()

    def compute_feature_intervals(self):
        """
        Groups contiguous detection seconds for each feature.
        If no standard feature (alert, mob, begging, softSong, rattle) is true,
        the second is grouped under "other".
        """
        feat_seconds = {feat: [] for feat in self.features}
        standard_feats = {"alert", "mob", "begging", "softSong", "rattle"}

        for det in self.detections:
            t = det["start_time"]
            has_feature = any(det.get(feat, False) for feat in standard_feats)
            if has_feature:
                for feat in standard_feats:
                    if det.get(feat, False):
                        feat_seconds[feat].append(t)
            else:
                feat_seconds["other"].append(t)

        intervals = {}
        for feat, times in feat_seconds.items():
            times = sorted(times)
            if not times:
                continue
            intervals[feat] = []
            start = times[0]
            prev = times[0]
            for current in times[1:]:
                if current - prev > 1.5:
                    intervals[feat].append((start, prev + 1))
                    start = current
                prev = current
            intervals[feat].append((start, prev + 1))
        return intervals

    def on_draw(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.ax_det.bbox)

    def toggle_play(self, event):
        if self.playing:
            sd.stop()
            self.playing = False
            self.play_button.label.set_text("Play")
        else:
            self.start_playback()

    def start_playback(self):
        start_sample = int(self.play_offset * self.sr)
        remaining_audio = self.audio[start_sample:]
        if len(remaining_audio) == 0:
            return
        sd.play(remaining_audio, self.sr)
        self.play_start_time = time.time()
        self.playing = True
        self.play_button.label.set_text("Pause")

    def update_playhead(self):
        if self.playing:
            elapsed = time.time() - self.play_start_time
            current_time = self.play_offset + elapsed
            if current_time >= self.total_duration:
                current_time = self.total_duration
                self.playing = False
                self.play_button.label.set_text("Play")
            self.playhead_line.set_xdata(current_time)
            self.fig.canvas.restore_region(self.background)
            self.ax_det.draw_artist(self.playhead_line)
            self.fig.canvas.blit(self.ax_det.bbox)

    def on_click(self, event):
        if event.inaxes == self.ax_det:
            new_time = event.xdata
            if new_time is None:
                return
            self.play_offset = new_time
            self.playhead_line.set_xdata(new_time)
            self.fig.canvas.restore_region(self.background)
            self.ax_det.draw_artist(self.playhead_line)
            self.fig.canvas.blit(self.ax_det.bbox)
            if self.playing:
                sd.stop()
                self.start_playback()

    def add_to_labeler(self, event):
        """Callback for the 'Add to Labeler' button for uncached files."""
        ensure_dataset(self.dataset_name, self.cache_base)
        dataset_labels_file = get_dataset_artifact_path(self.dataset_name, "labels.json", cache_base=self.cache_base)
        dataset_segments_file = get_dataset_artifact_path(self.dataset_name, "segments.json", cache_base=self.cache_base)
        dataset_audio_dir = get_dataset_import_path(self.dataset_name, "audio", cache_base=self.cache_base)
        dataset_embeddings_dir = get_dataset_import_path(self.dataset_name, "embeddings", cache_base=self.cache_base)

        os.makedirs(dataset_audio_dir, exist_ok=True)
        os.makedirs(dataset_embeddings_dir, exist_ok=True)

        # --- Determine suggested cluster (largest existing cluster + 1) ---
        suggestion = 1
        if os.path.exists(dataset_labels_file):
            try:
                with open(dataset_labels_file, "r") as f:
                    data = json.load(f)
                    clusters = [entry.get("cluster", 0) for entry in data.values()]
                    if clusters:
                        suggestion = max(clusters) + 1
            except Exception as e:
                print(f"Error reading {dataset_labels_file}: {e}")

        # Prompt the user via UI for a cluster number.
        root = tk.Tk()
        root.withdraw()
        cluster_int = simpledialog.askinteger("Cluster Input",
                                               f"Enter cluster number (suggested: {suggestion}):",
                                               initialvalue=suggestion,
                                               parent=root)
        root.destroy()
        if cluster_int is None:
            print("Cluster input cancelled. Operation aborted.")
            return

        # --- Generate GUID for the imported file ---
        short_guid = str(uuid.uuid4())[:8]

        # --- Update dataset labels.json using the new GUID for each detection ---
        if os.path.exists(dataset_labels_file):
            with open(dataset_labels_file, "r") as f:
                try:
                    labels = json.load(f)
                except json.JSONDecodeError:
                    labels = {}
        else:
            labels = {}

        for det in self.detections:
            if det['quality'] > 0:
                key = f"{short_guid}-{int(det['start_time'])}-{int(det['end_time'])}"
                det['cluster'] = cluster_int
                det['reviewed'] = False
                labels[key] = det
        with open(dataset_labels_file, "w") as f:
            json.dump(labels, f, indent=4)
        labels_count = len(self.detections)

        # --- Update dataset segments.json using the GUID as key ---
        if os.path.exists(dataset_segments_file):
            with open(dataset_segments_file, "r") as f:
                try:
                    segments = json.load(f)
                except json.JSONDecodeError:
                    segments = {}
        else:
            segments = {}

        segments_list = segments.get(short_guid, [])
        for det in self.detections:
            if det['quality'] > 0:
                segment = {
                    "common_name": "American Crow",
                    "scientific_name": "Corvus brachyrhynchos",
                    "start_time": det["start_time"],
                    "end_time": det["end_time"],
                    "confidence": 0,
                    "cluster": cluster_int
                }
                segments_list.append(segment)
        segments[short_guid] = segments_list
        with open(dataset_segments_file, "w") as f:
            json.dump(segments, f, indent=4)
        segments_count = len(segments_list)

        # --- File copy: copy the original raw file into the dataset imports directory ---
        dest_file = os.path.join(dataset_audio_dir, f"{short_guid}.mp3")
        if os.path.exists(self.source_file):
            try:
                shutil.copy(self.source_file, dest_file)
                file_copy_msg = f"File copied from '{self.source_file}' to '{dest_file}'."
            except Exception as e:
                file_copy_msg = f"Failed to copy file: {e}"
        else:
            file_copy_msg = f"Source file '{self.source_file}' not found. File not copied."

        # --- Save embeddings alongside the dataset import ---
        try:
            # Reload embeddings (from the raw file) using get_data.
            embeddings, volumes, audio, sr = get_data(self.source_file, self.public_path)
            emb_dest_file = os.path.join(dataset_embeddings_dir, f"{short_guid}.npy")
            np.save(emb_dest_file, embeddings)
            embeddings_msg = f"Embeddings saved to '{emb_dest_file}'."
        except Exception as e:
            embeddings_msg = f"Failed to save embeddings: {e}"

        # --- Summary Output ---
        print("\n=== Add to Labeler Summary ===")
        print(f"New File ID (GUID): {short_guid}")
        print(f"Cluster used: {cluster_int}")
        print(f"Labels added: {labels_count}")
        print(f"Segments added: {segments_count}")
        print(file_copy_msg)
        print(embeddings_msg)
        print("==============================\n")

        # --- Confirmation Popup ---
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Confirmation", "All detections successfully added to Labeler")
            root.destroy()
        except Exception as e:
            print(f"Confirmation popup failed: {e}")

###############################################################################
# Main entry point.
###############################################################################
def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect crow calls in a cached dataset file or raw audio file.")
    parser.add_argument("arg", nargs="?", help="File ID from a dataset, or a raw audio file path.")
    parser.add_argument("--dataset", default="all-public", help="Dataset to resolve cached file IDs from.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    parser.add_argument("--no-gui", action="store_true", help="Print detections and exit without opening the timeline UI.")
    args = parser.parse_args(argv)

    arg = args.arg or input("Enter file ID or file path: ").strip()

    # Determine if arg is an uncached file path.
    if os.path.isfile(arg):
        public_path = None
        raw_file = arg
    else:
        library_name = resolve_dataset_file_library(args.dataset, arg, cache_base=args.cache_dir)
        if library_name is None:
            raise FileNotFoundError(f"File ID {arg} was not found in dataset {args.dataset}")
        public_path = get_library_dir(library_name, args.cache_dir)
        raw_file = None
    detections, audio, sr = detect_file_segments(arg, public_path=public_path, include_audio=True)
    print(f"Found {len(detections)} detection segments.")
    print(json.dumps(detections, indent=4))

    description = generate_detection_summary(detections)
    print(f"Description:\n{description}\n")
    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f} seconds, Sample Rate: {sr} Hz")

    if args.no_gui:
        return

    label = os.path.basename(arg)
    player = TimelinePlayer(
        label,
        detections,
        audio,
        sr,
        public_path=public_path,
        raw_file=raw_file,
        dataset_name=args.dataset,
        cache_base=args.cache_dir,
    )
    plt.show()

if __name__ == "__main__":
    main()
