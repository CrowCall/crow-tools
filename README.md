# Decoding crow communication with AI

![crow-animation.gif](docs/images/crow-animation.gif)

This repo is built for processing crow audio data. It includes modules for embedding, classification, separation, labeling, downloading, denoising, segmentation, and analysis.

## Download and Prepare Data

Run this script to download, denoise, embed, and auto-label all crow audio files:

```
python get-data.py
```

## Structure
```
crow-sounds/
├── embed/
├── classifier/
├── separator/
├── labeler/
├── downloader/
├── denoiser/
├── segmenter/
├── analyzer/
└── .cache/          # Generated and downloaded data
```