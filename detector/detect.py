# !/usr/bin/env python3
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
from matplotlib.widgets import Button
from classifier.classify import predict_embedding

matplotlib.use("TkAgg")
player = None


def detect_file_segments(file_id, public_path=None, volume_threshold=0.0002, device=None):
    """
    Process a single file (given by file_id) and return a list of detection segments.
    Each detection corresponds to a one-second window where:
      - The volume (from a parallel volume file) exceeds `volume_threshold`
      - The classifier predicts a valid detection (quality > 0 and crowCount > 0)

    Args:
      file_id (str): the file identifier (used to load the embedding and volume files)
      public_path (str): base path to the .cache folder; if None, inferred relative to this file.
      volume_threshold (float): minimum volume required to run detection on a second.
      device (str): device for inference ("cuda" or "cpu"). If None, auto-detect.

    Returns:
      List[dict]: list of detection dictionaries.
    """
    if public_path is None:
        public_path = os.path.join(os.path.dirname(__file__), "..", ".cache")
    embeddings_dir = os.path.join(public_path, "embeddings-denoised")
    volumes_dir = os.path.join(public_path, "embeddings-denoised-volumes")

    embedding_path = os.path.join(embeddings_dir, f"{file_id}.npy")
    volume_path = os.path.join(volumes_dir, f"{file_id}.npy")

    if not os.path.exists(embedding_path):
        print(f"Embedding file not found for file ID {file_id}")
        return []
    if not os.path.exists(volume_path):
        print(f"Volume file not found for file ID {file_id}")
        return []

    embeddings = np.load(embedding_path)
    volume_data = np.load(volume_path)
    if volume_data.ndim > 1:
        volume_data = volume_data.squeeze(-1)

    detections = []
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    num_seconds = embeddings.shape[0]

    for i in range(num_seconds):
        if volume_data[i] <= volume_threshold:
            continue

        embedding = embeddings[i]
        pred = predict_embedding(embedding, device=device)

        # A valid detection: quality > 0 and crowCount > 0.
        if pred["quality"] > 0 and pred["crowCount"] > 0:
            detection = {
                "start_time": float(i),
                "end_time": float(i + 1),
                # Include classifier outputs.
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
    return detections


###############################################################################
# INTERACTIVE TIMELINE PLAYER
###############################################################################
class TimelinePlayer:
    def __init__(self, file_id, detections, audio, sr):
        """
        Args:
          file_id (str): the file ID.
          detections (list): list of detection dicts (per second).
          audio (np.ndarray): the audio waveform (mono).
          sr (int): sample rate.
        """
        self.file_id = file_id
        self.detections = detections
        self.audio = audio
        self.sr = sr
        self.total_duration = len(audio) / sr
        self.playing = False
        self.play_offset = 0.0  # in seconds; where playback starts
        self.play_start_time = None

        # Define features to plot (only boolean features are shown)
        self.features = ["alert", "mob", "begging", "softSong", "rattle"]
        self.feature_intervals = self.compute_feature_intervals()

        # Create the figure with two subplots (waveform on top, timeline below)
        self.fig, (self.ax_wave, self.ax_det) = plt.subplots(2, 1, sharex=True, figsize=(19.2, 8), dpi=100)
        # Fix the layout so that the detection axis remains stable.
        self.fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)
        self.fig.suptitle(f"File ID: {file_id} - Detection Timeline", fontsize=16)

        # Plot waveform in the top subplot.
        t = np.linspace(0, self.total_duration, len(self.audio))
        self.ax_wave.plot(t, self.audio, color="blue", lw=0.8)
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_xlim(0, self.total_duration)
        self.ax_wave.set_title("Waveform")

        # Plot detection timeline in the bottom subplot.
        self.ax_det.set_title("Detections")
        self.ax_det.set_ylim(0, len(self.features))
        self.ax_det.set_yticks([i + 0.5 for i in range(len(self.features))])
        self.ax_det.set_yticklabels(self.features)
        self.ax_det.set_xlabel("Time (s)")
        self.ax_det.set_xticks(np.linspace(0, self.total_duration, num=5))
        self.ax_det.set_xlim(0, self.total_duration)
        self.ax_det.set_autoscale_on(False)

        # Draw detection bars for each feature.
        colors = {"alert": "red", "mob": "green", "begging": "orange",
                  "softSong": "purple", "rattle": "brown"}
        for i, feat in enumerate(self.features):
            intervals = self.feature_intervals.get(feat, [])
            for (start, end) in intervals:
                self.ax_det.hlines(y=i + 0.5, xmin=start, xmax=end,
                                   colors=colors.get(feat, "gray"), linewidth=8)

        # Add a vertical playhead line (initially at 0).
        self.playhead_line = self.ax_det.axvline(x=0, color="black", linestyle="--", lw=2)

        # Hide playhead before caching background so that it isn't included.
        self.playhead_line.set_visible(False)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax_det.bbox)
        self.playhead_line.set_visible(True)

        # Add a Play/Pause button in a fixed position outside the main area.
        self.button_ax = self.fig.add_axes([0.87, 0.05, 0.1, 0.05])
        self.play_button = Button(self.button_ax, "Play")
        self.play_button.on_clicked(self.toggle_play)

        # Connect mouse click on the detection axis.
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        # Update cached background on every draw event.
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)

        # Use a timer with a moderate interval.
        self.timer = self.fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.update_playhead)
        self.timer.start()

    def on_draw(self, event):
        # Update the cached background whenever the figure is drawn.
        self.background = self.fig.canvas.copy_from_bbox(self.ax_det.bbox)

    def compute_feature_intervals(self):
        """
        Group contiguous seconds for each feature into intervals.
        Returns:
          dict: mapping feature name -> list of (start, end) intervals.
        """
        intervals = {feat: [] for feat in self.features}
        feat_seconds = {feat: [] for feat in self.features}
        for det in self.detections:
            t = det["start_time"]
            for feat in self.features:
                if det.get(feat, False):
                    feat_seconds[feat].append(t)
        for feat, times in feat_seconds.items():
            times = sorted(times)
            if not times:
                continue
            start = times[0]
            prev = times[0]
            for current in times[1:]:
                if current - prev > 1.5:
                    intervals[feat].append((start, prev + 1))
                    start = current
                prev = current
            intervals[feat].append((start, prev + 1))
        return intervals

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


def main():
    if len(sys.argv) < 2:
        file_id = input("Enter file ID: ").strip()
    else:
        file_id = sys.argv[1]

    # Assume the audio files are stored in .cache/library as MP3s.
    base_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
    library_dir = os.path.join(base_dir, "library")
    audio_path = os.path.join(library_dir, f"{file_id}.mp3")
    if not os.path.exists(audio_path):
        print(f"Audio file not found for file ID {file_id}")
        sys.exit(1)

    print(f"Processing detections for file ID {file_id} ...")
    detections = detect_file_segments(file_id, public_path=base_dir)
    print(f"Found {len(detections)} detection segments.")
    print(json.dumps(detections, indent=4))

    print("Loading audio ...")
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    duration = len(audio) / sr
    print(f"Audio duration: {duration:.2f} seconds, Sample Rate: {sr} Hz")

    # Create the interactive timeline player.
    player = TimelinePlayer(file_id, detections, audio, sr)
    plt.show()


if __name__ == "__main__":
    # Good test file ids
    # - 609381653 (alert: 17, rattle: 4)
    # - 490021141 (alert: 12, mob: 5, begging: 6, softSong: 2, rattle: 5)
    # - 587709701 (mob: 1, softSong: 5, rattle: 5)
    # - 539582961 (begging: 4, softSong: 1, rattle: 1)
    main()
