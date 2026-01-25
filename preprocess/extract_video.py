# (768,)
import cv2
import torch
import numpy as np
from transformers import ViTModel, ViTImageProcessor

class ViTVideoExtractor:

    def __init__(self,
                 model_name="google/vit-base-patch16-224"):

        print("Loading ViT...")

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        self.model.eval()


    def sample_frames(self, video_path, max_frames=8):

        cap = cv2.VideoCapture(video_path)

        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(total // max_frames, 1)

        idx = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if idx % step == 0:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            idx += 1

            if len(frames) >= max_frames:
                break

        cap.release()

        return frames


    def encode(self, video_path):

        frames = self.sample_frames(video_path)

        if len(frames) == 0:
            return np.zeros(768)

        inputs = self.processor(
            frames,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = self.model(**inputs)

        cls = out.last_hidden_state[:,0,:]   # CLS tokens

        return cls.mean(0).numpy()
