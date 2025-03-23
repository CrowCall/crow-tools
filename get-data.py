from downloader.download_backgrounds import start_downloads as start_downloads_background
from downloader.download_crows_ebird import start_downloads as start_downloads_ebird
from downloader.download_crows_xeno import start_downloads as start_downloads_xeno

# Download all crow audio, and mix-related audio
for download in [start_downloads_background, start_downloads_ebird, start_downloads_xeno]:
    download()
