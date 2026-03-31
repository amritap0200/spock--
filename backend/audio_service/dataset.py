import os
import soundfile as sf
from scipy.signal import resample
import numpy as np
import torch
from python_speech_features import logfbank
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        audio, sr = sf.read(path)

        # convert to mono if needed
        if len(audio.shape) > 1:
           audio = audio.mean(axis=1)

        # resample if needed
        if sr != 16000:
           num_samples = int(len(audio) * 16000 / sr)
           audio = resample(audio, num_samples)
           sr = 16000

        mel = logfbank(
           audio,
           samplerate=sr,
           nfilt=128,
           winlen=1024/sr,
           winstep=512/sr,
           nfft=1024
        )

        mel = mel.T

        mel = np.nan_to_num(mel)

        MAX_LEN = 300  # choose fixed time dimension

        mel = torch.tensor(mel).float()

        # Pad or trim time dimension
        if mel.shape[1] < MAX_LEN:
            pad_size = MAX_LEN - mel.shape[1]
            mel = torch.nn.functional.pad(mel, (0, pad_size))
        else:
            mel = mel[:, :MAX_LEN]

        mel = mel.unsqueeze(0)  # (1, 128, 300)

        return mel, torch.tensor(label).float()
