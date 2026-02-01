# EAC-Agent (Pre-Process)
This project preprocesses raw IEMOCAP and MELD datasets and generates
serialized multimodal feature files (`.pkl`) using:

- GloVe for text embeddings
- MFCC + GMM for audio modeling
- Vision Transformer (ViT) for visual features

The generated `.pkl` files are later used for training the EAC-Agent model.

## Project Structure
```console
EAC-Preprocess/
│
├── data/
│ ├── IEMOCAP/ # Raw IEMOCAP dataset
│ ├── MELD/ # Raw MELD dataset
│ └── processed/ # Output PKL files
│
├── preprocess/
│ ├── extract_text.py
│ ├── extract_audio.py
│ ├── extract_video.py
│ ├── build_pkl_iemocap.py
│ └── build_pkl_meld.py
│ └── utils.py
│
├── glove/
│ └── glove.6B.300d.txt
│
└── requirements.txt
```
## Dataset Download
### IEMOCAP
Request access from:
https://sail.usc.edu/iemocap/

After approval, download and extract into:

data/IEMOCAP

### MELD
Download from:
https://affective-meld.github.io/

Extract into:

data/MELD

## Download GloVe Embeddings
Download pretrained GloVe vectors:

```console
mkdir glove
cd glove

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
```
Ensure that the following file exists:

glove/glove.6B.300d.txt

## Environment Setup
Create and activate a virtual environment

```console
python3 -m venv venv
source venv/bin/activate
```
Install dependencies
```console
pip install -r requirements.txt
```
## Generate Feature Files
Run preprocessing scripts
```console
cd preprocess

python build_pkl_iemocap.py
python build_pkl_meld.py
```
