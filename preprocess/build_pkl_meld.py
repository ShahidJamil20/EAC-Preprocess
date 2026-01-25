import os
import glob
import pickle
import pandas as pd
from tqdm import tqdm

from extract_text import GloveTextExtractor
from extract_audio import GMMAudioExtractor
from extract_video import ViTVideoExtractor

MELD_ROOT = "data/MELD"
SAVE_PATH = "data/processed/meld_multimodal_features.pkl"

# ------------------------------------------------
def main():
    print("Initializing extractors...")
    text_ext = GloveTextExtractor()
    audio_ext = GMMAudioExtractor()
    video_ext = ViTVideoExtractor()

    print("Collecting audio for GMM...")

    wav_files = glob.glob(
        f"{MELD_ROOT}/**/*_audio/*.wav",
        recursive=True
    )

    if audio_ext.gmm is None:
        audio_ext.train_gmm(wav_files)


    data = {
        "text": {},
        "audio": {},
        "video": {},
        "labels": {},
        "train_ids": [],
        "dev_ids": [],
        "test_ids": []
    }

    print("Extracting features...")

    for split in ["train", "dev", "test"]:

        csv_path = f"{MELD_ROOT}/{split}_sent_emo.csv"

        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df)):

            utt = str(row["Utterance_ID"])
            text = row["Utterance"]
            label = row["Emotion"]

            wav = f"{MELD_ROOT}/{split}_audio/{utt}.wav"
            vid = f"{MELD_ROOT}/{split}_video/{utt}.mp4"

            if not os.path.exists(wav):
                continue

            if not os.path.exists(vid):
                continue

            t = text_ext.encode(text)
            a = audio_ext.encode(wav)
            v = video_ext.encode(vid)

            data["text"][utt] = t
            data["audio"][utt] = a
            data["video"][utt] = v
            data["labels"][utt] = label

            data[f"{split}_ids"].append(utt)

    os.makedirs("data/processed", exist_ok=True)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Saved:", SAVE_PATH)


# --------------------

if __name__ == "__main__":
    main()
