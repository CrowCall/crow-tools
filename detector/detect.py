#!/usr/bin/env python3
import json
import os
import sys
import time

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch

from classifier.classify import predict_embedding
from embedder.embed import generate_embeddings

def is_headless():
    return not os.environ.get("DISPLAY") and os.name != "nt"
if is_headless():
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

###############################################################################
# Helper function to retrieve embeddings, volumes, audio and sample rate.
###############################################################################
def get_data(arg, public_path=None):
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
      - {public_path}/library/{file_id}.mp3
    """
    if public_path is None:
        public_path = os.path.join(os.path.dirname(__file__), "..", ".cache")

    if os.path.isfile(arg):
        # Uncached mode: arg is a file path.
        print(f"Processing uncached file: {arg}")
        sample_rate = 8000  # force consistent sample rate (as in embed_all.py)
        try:
            audio, sr = librosa.load(arg, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio from {arg}: {e}")
            sys.exit(1)
        # Generate raw embeddings (shape: [num_frames, feature_dim])
        print("Generating embeddings ...")
        full_embeddings = generate_embeddings(audio)
        full_embeddings = np.array(full_embeddings, dtype=np.float32)
        num_frames = full_embeddings.shape[0]
        # Average every 25 frames to get one embedding per second.
        chunk_size = 25
        num_chunks = int(np.ceil(num_frames / chunk_size))
        embedding_means = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_frames)
            chunk = full_embeddings[start_idx:end_idx]
            mean_emb = np.mean(chunk, axis=0)
            embedding_means.append(mean_emb)
        embeddings = np.stack(embedding_means, axis=0)

        # Compute per-second volume metrics (mean absolute amplitude).
        total_samples = len(audio)
        total_seconds = int(np.ceil(total_samples / sr))
        volumes = []
        for sec in range(total_seconds):
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
        library_dir = os.path.join(public_path, "library")
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
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            print(f"Error loading audio: {e}")
            sys.exit(1)
        return embeddings, volumes, audio, sr

###############################################################################
# Detection function (common for both cached and uncached data)
###############################################################################
def detect_file_segments(arg, volume_threshold=0.0002, device=None, public_path=None):
    """
    Loads embeddings and volume data (and audio) using get_data() and then runs
    the classifier on each second where the volume exceeds the threshold.
    Returns:
      - detections: list of detection dictionaries
      - audio: the audio waveform
      - sr: sample rate
    """
    embeddings, volumes, audio, sr = get_data(arg, public_path)
    detections = []
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Process seconds based on the lesser of embeddings length and volume length.
    num_seconds = min(embeddings.shape[0], len(volumes))
    for i in range(num_seconds):
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
# INTERACTIVE TIMELINE PLAYER (unchanged)
###############################################################################
class TimelinePlayer:
    def __init__(self, label, detections, audio, sr):
        """
        Args:
          label (str): the file identifier or filename label.
          detections (list): list of detection dicts (per second).
          audio (np.ndarray): the audio waveform (mono).
          sr (int): sample rate.
        """
        self.label = label
        self.detections = detections
        self.audio = audio
        self.sr = sr
        self.total_duration = len(audio) / sr
        self.playing = False
        self.play_offset = 0.0  # seconds; playback start position
        self.play_start_time = None

        # Define the boolean features to display.
        # "other" is added for visualization if no standard attribute is detected.
        self.features = ["alert", "mob", "begging", "softSong", "rattle", "other"]
        self.feature_intervals = self.compute_feature_intervals()

        # Create a figure with two subplots: waveform and detection timeline.
        self.fig, (self.ax_wave, self.ax_det) = plt.subplots(2, 1, sharex=True, figsize=(19.2, 8), dpi=100)
        self.fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)
        self.fig.suptitle(f"File: {label} - Detection Timeline", fontsize=16)

        # Plot waveform.
        t = np.linspace(0, self.total_duration, len(self.audio))
        self.ax_wave.plot(t, self.audio, color="blue", lw=0.8)
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
        For each detection, if none of the standard features (alert, mob, begging,
        softSong, rattle) are true, then that second is grouped under "other".
        """
        # Initialize lists for each feature.
        feat_seconds = {feat: [] for feat in self.features}
        standard_feats = {"alert", "mob", "begging", "softSong", "rattle"}

        for det in self.detections:
            t = det["start_time"]
            # Determine if any standard feature is true.
            has_feature = any(det.get(feat, False) for feat in standard_feats)
            if has_feature:
                for feat in standard_feats:
                    if det.get(feat, False):
                        feat_seconds[feat].append(t)
            else:
                # If no standard feature is present, add to "other".
                feat_seconds["other"].append(t)

        # Group contiguous seconds into intervals.
        intervals = {}
        for feat, times in feat_seconds.items():
            times = sorted(times)
            if not times:
                continue
            intervals[feat] = []
            start = times[0]
            prev = times[0]
            for current in times[1:]:
                if current - prev > 1.5:  # gap indicates a new interval
                    intervals[feat].append((start, prev + 1))
                    start = current
                prev = current
            intervals[feat].append((start, prev + 1))
        return intervals

    # (Keep the rest of your TimelinePlayer methods unchanged)
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

###############################################################################
# Main entry point.
###############################################################################
def main():
    if len(sys.argv) < 2:
        arg = input("Enter file ID or file path: ").strip()
    else:
        arg = sys.argv[1]

    # Use the .cache directory relative to this script.
    public_path = os.path.join(os.path.dirname(__file__), "..", ".cache")
    detections, audio, sr = detect_file_segments(arg, public_path=public_path)
    print(f"Found {len(detections)} detection segments.")
    print(json.dumps(detections, indent=4))

    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f} seconds, Sample Rate: {sr} Hz")

    # Use the basename (file id or file name) as a label.
    label = os.path.basename(arg)
    player = TimelinePlayer(label, detections, audio, sr)
    plt.show()

if __name__ == "__main__":
    main()
