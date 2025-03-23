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
