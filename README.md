# Decoding Crow Communication with AI  

## Crows are among the most intelligent and vocal birds on the planet. But what exactly are they saying? 

<img src="docs/images/crow-animation.gif" align="left" width="50%" title="Original animation by @owlmaddie">

This project aims to **explore** and **decode** the rich and mysterious world of **crow communication** using cutting-edge 
machine learning and audio analysis.

This repository is your toolkit for working with thousands of **American crow** ([Corvus brachyrhynchos](https://www.allaboutbirds.org/guide/American_Crow/overview)) **calls**—from downloading and cleaning audio, to 
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
sudo apt install ffmpeg git-lfs portaudio19-dev python3-tk
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

The downloader module retrieves a large collection of crow vocalizations (13+ GB) from multiple public repositories. 
It handles the complexities of connecting to each source, downloading the audio files, and storing relevant metadata 
for proper attribution. Credits and licensing info for all files are saved in the `.cache/csv/` directory.

Crow-tools relies on openly available datasets for research and development. We gratefully acknowledge the following sources:

- [Macaulay Library – American Crow (*Corvus brachyrhynchos*)](https://search.macaulaylibrary.org/catalog?taxonCode=amecro&mediaType=audio)  
  © Cornell Lab of Ornithology. A comprehensive archive of wildlife recordings, used in accordance with licensing terms for non-commercial research.

- [Xeno-Canto – American Crow (*Corvus brachyrhynchos*)](https://xeno-canto.org/species/corvus-brachyrhynchos)  
  A global, community-powered collection of bird calls, shared under various Creative Commons licenses. Many thanks to the recordists who make this work possible.

### Denoiser
The denoiser module cleans the crow audio files by removing unwanted background noise. This process improves the 
quality of the audio for subsequent processing steps by focusing on the relevant crow sounds. It also enables
the creation of mixes (overlapping crow sounds) to train our separator model. This module utilizes the [biodenoising](https://github.com/earthspecies/biodenoising-inference)
module created by [Earth Species Project](https://earthspecies.org/).

![denoiser.png](docs/images/denoiser.png)

### Classifier
The classifier module analyzes crow call embeddings and categorizes them (i.e. auto labels) into various types such as 
alert, number of calls, age indicators, rattles, soft songs, and quality of audio. It processes the embedded data and 
applies machine learning techniques to identify and label crow vocalizations.

```python
    {
      "crowCount": int in [1,2,3,4],   # 1 = single, 2 = two crows, 3 = unused, 4 = crowd
      "crowAge": int in [1,2],         # 1 = adult, 2 = juvenile
      "alert": bool,                   # alert calls
      "begging": bool,                 # food related calls
      "softSong": bool,                # sub songs | soft sounds
      "rattle": bool,                  # rattle sounds
      "mob": bool,                     # anger calls | mob | attack
      "quality": int in [1,2,3],       # 1 = bad, 2 = good, 3 = unused
    }
```

### Detector
The detector module leverages our custom trained crow classifier, to quickly find all crow sounds across an audio file.
By isolating these segments, the module enables more focused analysis and processing of individual crow calls and vocalizations.
We also include an interactive crow timeline app to review and listen to the detections:

![detector-timeline.png](docs/images/detector-timeline.png)

```json
[
    {
        "start_time": 37.0,
        "end_time": 38.0,
        "crowCount": 2,
        "crowAge": 1,
        "alert": false,
        "begging": false,
        "softSong": false,
        "rattle": true,
        "mob": false,
        "quality": 2
    }
]
```

### Embedder
The embedder module transforms each crow call into a 768-dimensional vector using the [AVES](https://github.com/earthspecies/aves?tab=readme-ov-file#birdaves) embedding model. This 
transformation creates a numerical representation of the audio, which is essential for further analysis and machine 
learning applications.

![embeddings.gif](docs/videos/embeddings.gif)

### Labeler
The labeler module provides a web interface for manual labeling of crow calls. This interface is designed for 
human labeling and review, ensuring that the training data for the classifier is accurate and reliable. It also
provides a 3D interactive embedding feature. This web app is created with Vue v3 and Node.js.

![labeler.png](docs/images/labeler.png)

### Separator
The separator module is responsible for separating overlapping crow calls into distinct audio files. This process 
enables clearer analysis by isolating individual calls that may be mixed together in the original recordings.

![separator.png](docs/images/separator.png)

## Directory Structure
```
crow-tools/
├── classifier/  # classify types of crow calls (alert, count, age, rattle, soft/sub song, quality)
├── denoiser/    # denoise crow audio files (remove background noises with biodenoising model)
├── detector/    # detect crow audio segments (1 second each, uses classifier model)
├── downloader/  # download library of crow audio files
├── embedder/    # embed crow calls into 768 dimensions (AVES embedding model)
├── labeler/     # human labeling web app (for training classifier)
├── separator/   # separate overlapping crow calls into seaparate audio files (train and inference)
└── .cache/      # all downloaded and generated files (30+ GB)
```

## Authors

This project is a collaborative effort driven by a shared curiosity and love for crows — with the ultimate goal of better understanding their complex and intelligent vocal language.

- **[Jonathan Thomas](mailto:crows@openshot.org)** is the creator of [OpenShot Video Editor](https://www.openshot.org/) and brings deep experience in software development and artificial intelligence. Through his company, **OpenShot Studios LLC**, he leads the technical development of tools for large-scale audio analysis, machine learning, and crow call classification.

- **[Madeline Thomas](mailto:crows@owlmaddie.com) (@owlmaddie)** is a professional [artist / animator](https://www.owlmaddie.com/) and crow enthusiast. Through **owlmaddie LLC**, she contributes her creative talents to the project — labeling thousands of crow calls, designing expressive crow animations, and helping to communicate the science through visual storytelling.

Together, we're building a **state-of-the-art toolkit for decoding and exploring crow communication**, blending AI, design, and a shared passion for one of nature’s most fascinating birds.

## Citation

Give our crows a shout-out — cite `crow-tools` in your work!

GitHub provides a citation file — just click the **Cite this repository** button in the sidebar at the top of this page.

