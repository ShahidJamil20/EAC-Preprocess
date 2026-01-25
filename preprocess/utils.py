import os
import pickle

def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
