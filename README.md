# Decoding crow communication with AI

![crow-animation.gif](docs/images/crow-animation.gif)

This repo is built for processing crow audio data. It includes modules for embedding, classification, separation, labeling, downloading, denoising, segmentation, and analysis.

## Download and Prepare Data

Run the main pipeline script to download, denoise, embed, auto-label, and get a summary:

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