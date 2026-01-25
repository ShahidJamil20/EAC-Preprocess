import librosa
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import os

class GMMAudioExtractor:
    def __init__(self,
                 n_components=16,
                 sr=16000,
                 gmm_path="preprocess/models/audio_gmm.pkl"):

        self.sr = sr
        self.n_components = n_components
        self.gmm_path = gmm_path

        if os.path.exists(gmm_path):
            print("Loading trained GMM...")
            with open(gmm_path, "rb") as f:
                self.gmm = pickle.load(f)
        else:
            self.gmm = None


    def extract_mfcc(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40,
            hop_length=160,
            n_fft=400
        )

        return mfcc.T   # (frames, dim)


    def train_gmm(self, wav_files):
        print("Training GMM...")
        feats = []

        for wav in wav_files:

            mfcc = self.extract_mfcc(wav)
            feats.append(mfcc)

        feats = np.vstack(feats)

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=200,
            random_state=0
        )

        self.gmm.fit(feats)

        os.makedirs(os.path.dirname(self.gmm_path), exist_ok=True)

        with open(self.gmm_path, "wb") as f:
            pickle.dump(self.gmm, f)

        print("GMM saved:", self.gmm_path)


    def encode(self, wav_path):
        if self.gmm is None:
            raise RuntimeError("Train GMM first!")

        mfcc = self.extract_mfcc(wav_path)
        post = self.gmm.predict_proba(mfcc)
        means = self.gmm.means_
        covs = self.gmm.covariances_

        # Weighted mean/var
        w_mean = post.T @ mfcc / (np.sum(post, axis=0)[:,None]+1e-8)

        w_var = covs

        return np.concatenate([
            w_mean.flatten(),
            w_var.flatten()
        ])
