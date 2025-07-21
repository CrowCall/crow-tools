import sys

print("""
======================================================================
                            CrowTools 
======================================================================
CrowTools is ready to take flight and process *tons* of crow audio data!
It needs over *30 GB* of space and hours to caw-mplete (scales with %)!
Each file is downloaded, denoised, segmented, embedded, and classified.

Previously processed files? We’ll hop past them like a crafty crow!
----------------------------------------------------------------------
Enter % of data you want to download? (1-100)
  Try 1 for a quick test, 100 for everything!
----------------------------------------------------------------------
""")
response = input("> ").strip()
try:
    percentage = int(response)
    if not 1 <= percentage <= 100:
        raise ValueError
except ValueError:
    print("Invalid input. Please enter a number between 1 and 100.")
    sys.exit(1)

from downloader.download_backgrounds import start_downloads as start_downloads_background
from downloader.download_crows_ebird import start_downloads as start_downloads_ebird
from downloader.download_crows_xeno import start_downloads as start_downloads_xeno
from denoiser.denoise_crows import start_denoising
from detector.detect_all import start_detections
from embedder.embed_all import start_embeddings

# Download all crow audio and mix-related audio
for download in [start_downloads_background, start_downloads_ebird, start_downloads_xeno]:
    download(percent=percentage)

# Denoise all crow audio
start_denoising()

# Embed all crow audio (both original and denoised)
start_embeddings(denoised=False)
start_embeddings(denoised=True)

# Detect 1 second crow segments (uses classifier + embeddings)
# This also creates the auto_labels.json file
start_detections()
