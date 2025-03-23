import sys

print("""
This script will download and process a large amount of crow audio data.
It requires over 30 GB of disk space and will take several hours to complete.
Each audio file will be downloaded, denoised, segmented, embedded, and classified.

Files that are already processed will be skipped.
Are you ready to begin? (y/n)
""")
response = input("> ").strip().lower()
if response not in ("y", "yes"):
    print("Exiting. Please run the script again when you're ready.")
    sys.exit(0)

from downloader.download_backgrounds import start_downloads as start_downloads_background
from downloader.download_crows_ebird import start_downloads as start_downloads_ebird
from downloader.download_crows_xeno import start_downloads as start_downloads_xeno
from denoiser.denoise_crows import start_denoising
from detector.detect_segments import start_detections
from classifier.auto_label import start_labeling
from embedder.embed_all import start_embeddings

# Download all crow audio and mix-related audio
for download in [start_downloads_background, start_downloads_ebird, start_downloads_xeno]:
    download()

# Denoise all crow audio
start_denoising()

# Detect 3 second crow segments (using BirdNET)
start_detections()

# Embed all detected segments
start_embeddings()

# Auto-Label all segments (using classifier)
start_labeling()
