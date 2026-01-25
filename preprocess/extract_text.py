import numpy as np
import re
from gensim.models import KeyedVectors

class GloveTextExtractor:
    def __init__(self, glove_path="preprocess/glove/glove.6B.300d.txt"):
        print("Loading GloVe...")
        self.model = KeyedVectors.load_word2vec_format(
            glove_path,
            binary=False,
            no_header=True
        )
        self.dim = 300

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    def encode(self, text):
        text = self.clean_text(text)

        tokens = text.split()

        vectors = []

        for w in tokens:
            if w in self.model:
                vectors.append(self.model[w])

        # No known words
        if len(vectors) == 0:
            return np.zeros(self.dim)

        # Average pooling
        return np.mean(vectors, axis=0)
