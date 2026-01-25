from extract_audio import GMMAudioExtractor
import glob

wav_files = glob.glob("data/IEMOCAP/**/wav/*.wav", recursive=True)

ext = GMMAudioExtractor()
ext.train_gmm(wav_files)
