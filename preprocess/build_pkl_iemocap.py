import os
import glob
import pickle
from tqdm import tqdm

from extract_text import GloveTextExtractor
from extract_audio import GMMAudioExtractor
from extract_video import ViTVideoExtractor


IEMOCAP_ROOT = "data/IEMOCAP"
SAVE_PATH = "data/processed/iemocap_multimodal_features.pkl"


# -----------------
# Load Emotion Labels
# ---------------

def load_labels():

    label_map = {
        "ang": 0,
        "hap": 1,
        "exc": 1,
        "sad": 2,
        "neu": 3,
        "fru": 4,
        "fea": 5,
        "sur": 6,
        "dis": 7
    }

    labels = {}

    for ses in range(1, 6):

        lab_dir = f"{IEMOCAP_ROOT}/Session{ses}/dialog/EmoEvaluation"

        for file in os.listdir(lab_dir):

            if not file.endswith(".txt"):
                continue

            path = os.path.join(lab_dir, file)

            with open(path) as f:
                for line in f:

                    if line.startswith("["):

                        parts = line.split()

                        utt = parts[3]
                        emo = parts[4]

                        if emo in label_map:
                            labels[utt] = label_map[emo]

    return labels


# ------
# Main
# ----

def main():

    print("Initializing extractors...")

    text_ext = GloveTextExtractor()
    audio_ext = GMMAudioExtractor()
    video_ext = ViTVideoExtractor()

    print("Loading emotion labels...")
    emo_labels = load_labels()

    print("Collecting audio for GMM training...")

    wav_files = glob.glob(
        f"{IEMOCAP_ROOT}/Session*/sentences/wav/*.wav",
        recursive=True
    )

    if audio_ext.gmm is None:
        audio_ext.train_gmm(wav_files)

    data = {
        "text": {},
        "audio": {},
        "video": {},
        "labels": {},
        "speakers": {},
        "train_ids": [],
        "test_ids": []
    }

    print("Extracting features...")

    for ses in range(1, 6):

        ses_path = f"{IEMOCAP_ROOT}/Session{ses}"

        trans_dir = f"{ses_path}/dialog/transcriptions"

        for file in os.listdir(trans_dir):

            path = os.path.join(trans_dir, file)

            with open(path) as f:
                lines = f.readlines()

            for line in tqdm(lines):

                if ":" not in line:
                    continue

                utt, text = line.split(":", 1)

                utt = utt.strip()

                wav = f"{ses_path}/sentences/wav/{utt}.wav"
                vid = f"{ses_path}/video/{utt}.avi"

                if not os.path.exists(wav):
                    continue

                if not os.path.exists(vid):
                    continue

                t_feat = text_ext.encode(text)
                a_feat = audio_ext.encode(wav)
                v_feat = video_ext.encode(vid)

                spk = utt.split("_")[0][-1]  # F / M

                spk = 0 if spk == "F" else 1

                if utt not in emo_labels:
                    continue

                label = emo_labels[utt]

                data["text"][utt] = t_feat
                data["audio"][utt] = a_feat
                data["video"][utt] = v_feat
                data["labels"][utt] = label
                data["speakers"][utt] = spk

                # Split: Session5 = test
                if ses == 5:
                    data["test_ids"].append(utt)
                else:
                    data["train_ids"].append(utt)


    os.makedirs("data/processed", exist_ok=True)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Saved:", SAVE_PATH)


# ------------------------------------------------

if __name__ == "__main__":
    main()
