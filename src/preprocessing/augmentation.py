"""
Augmentación de datos sobre el Mel-spectrograma (SpecAugment).
Casi obligatorio dado el tamaño reducido del dataset (200 audios).
"""

import random

import torch
import torchaudio.transforms as T


class SpecAugment:
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 20, p: float = 0.5):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (tiempo, n_mels) -> transforms de torchaudio esperan (..., freq, tiempo)
        x = mel.transpose(0, 1).unsqueeze(0)  # (1, n_mels, tiempo)

        if random.random() < self.p:
            x = self.freq_mask(x)
        if random.random() < self.p:
            x = self.time_mask(x)

        return x.squeeze(0).transpose(0, 1)  # de vuelta a (tiempo, n_mels)
