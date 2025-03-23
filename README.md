# Decoding crow communication with AI

![crow-animation.gif](docs/images/crow-animation.gif)

This repo is built for processing crow audio data. It includes modules for embedding, classification, separation, labeling, downloading, denoising, segmentation, and analysis.

## Install Dependencies

```
pip install -r requirements.txt
```

## Download and Prepare Data

Run this script to download, denoise, embed, and auto-label all crow audio files. 
NOTE: This will download more than **30 GB** of data into a local `.cache` directory.

```
python get-data.py
```

## Directory Structure
```
crow-sounds/
├── classifier/  # classify types of crow calls
├── denoiser/    # denoise crow audio files
├── detector/    # detect 3 second segments of crow calls (BirdNET)
├── docs/        # documentation
├── downloader/  # download library of crow calls
├── embedder/    # embed crow calls into 768 dimensions (AVES)
├── labeler/     # human labeling web app
├── separator/   # separate overlapping crow calls into seaparate files
└── .cache/      # all downloaded and generated audio files
```