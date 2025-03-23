# Decoding Crow Communication with AI  

## Crows are among the most intelligent and vocal birds on the planet. But what exactly are they saying? 

<img src="docs/images/crow-animation.gif" align="left" width="50%">

This project aims to **explore** and **decode** the rich and mysterious world of **crow communication** using cutting-edge 
machine learning and audio analysis.

This repository is your toolkit for working with thousands of crow calls—from downloading and cleaning audio, to 
detecting individual calls, generating embeddings, classifying vocalizations, and even manually labeling them 
through a web interface.

Whether you're a bird enthusiast, a researcher, or just someone fascinated by animal intelligence, this project 
offers a glimpse into the complex vocal language of these remarkable creatures—and the possibility of better 
understanding them through technology.

## Install Dependencies

This project was built on **Ubuntu 24.04** and **Python 3.8+**, however it should be compatible with most linux
and mac systems. **Git LFS** is required to correctly clone, pull, and inflate all models. **FFmpeg** is required
to run the denoiser.

```
sudo apt install ffmpeg git-lfs
pip install -r requirements.txt
```

## Download and Prepare Data

Run this script to download, denoise, embed, and auto-label all crow audio files. 
NOTE: This will download more than **30 GB** of data into a local `.cache` directory.

```
python get-data.py
```

## Module Overview

### Downloader
The downloader module retrieves a vast library of crow audio files from multiple sources (13+ GB). It manages the 
complexities of connecting to various data repositories, ensuring that all necessary audio files are collected and stored 
efficiently. Credits and metadata are located in the `.cache/csv/` folder.

### Denoiser
The denoiser module cleans the crow audio files by removing unwanted background noise. This process improves the 
quality of the audio for subsequent processing steps by focusing on the relevant crow sounds. It also enables
the creation of mixes (overlapping crow sounds) to train our separator.

![denoiser.png](docs/images/denoiser.png)

### Detector
The detector module leverages BirdNET to identify and extract 3-second segments of crow calls from longer audio 
recordings. By isolating these segments, the module enables more focused analysis and processing of individual 
crow calls.

```json
    { 
      "common_name": "American Crow", 
      "scientific_name": "Corvus brachyrhynchos", 
      "start_time": 3.0, 
      "end_time": 6.0, 
      "confidence": 0.898
    }
```

### Embedder
The embedder module transforms each crow call into a 768-dimensional vector using the AVES embedding model. This 
transformation creates a numerical representation of the audio, which is essential for further analysis and machine 
learning applications.

![embeddings.png](docs/images/embeddings.png)

### Labeler
The labeler module provides a web interface for manual labeling of crow calls. This interface is designed for 
human labeling and review, ensuring that the training data for the classifier is accurate and reliable. It also
provides a 3D interactive embedding feature. This web app is created with Vue v3 and Node.js.

![labeler.png](docs/images/labeler.png)

### Classifier
The classifier module analyzes crow call embeddings and categorizes them (i.e. auto labels) into various types such as 
alert, number of calls, age indicators, rattles, soft songs, and quality of audio. It processes the embedded data and 
applies machine learning techniques to identify and label crow vocalizations.

```python
    {
      "crowCount": int in [1,2,3,4],   # 1 = single, 2 = two crows, 3 = three crows, 4 = crowd
      "crowAge": int in [1,2],         # 1 = adult, 2 = juvenile
      "alert": bool,                   # attack | alert | mob
      "begging": bool,                 # food related calls
      "grief": bool,                   # sadness | grief
      "softSong": bool,                # songs | soft sounds
      "rattle": bool,                  # rattle sounds
      "quality": int in [1,2,3],       # 1 = bad/low, 2 = average, 3 = high quality
      "reviewed": bool                 # human reviewed (else auto labeled)
    }
```

### Separator
The separator module is responsible for separating overlapping crow calls into distinct audio files. This process 
enables clearer analysis by isolating individual calls that may be mixed together in the original recordings.

![separator.png](docs/images/separator.png)

## Directory Structure
```
crow-tools/
├── classifier/  # classify types of crow calls (alert, count, age, rattle, soft song, bad quality)
├── denoiser/    # denoise crow audio files (remove background noises)
├── detector/    # detect segments (3 seconds each) of crow calls (BirdNET)
├── downloader/  # download library of crow audio files
├── embedder/    # embed crow calls into 768 dimensions (AVES embedding model)
├── labeler/     # human labeling web app (for training the classifier)
├── separator/   # separate overlapping crow calls into seaparate files (train and inference)
└── .cache/      # all downloaded and generated files (30+ GB)
```

