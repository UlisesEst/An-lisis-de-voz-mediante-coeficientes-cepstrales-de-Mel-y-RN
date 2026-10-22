"""
Fine-tuning de un Wav2Vec2 preentrenado como comparación contra el
Transformer entrenado desde cero.

Nota: este modelo trabaja sobre la forma de onda cruda, NO sobre el
Mel-spectrograma -- Wav2Vec2 aprende su propia representación interna.
Se necesita un dataset/collate_fn distinto que devuelva waveform crudo
en vez de Mel (pendiente en datasets/, se agrega en la siguiente fase).
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(MODEL_NAME)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.clasificador = nn.Linear(hidden_size, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (batch, samples)
        """
        out = self.encoder(waveform).last_hidden_state  # (batch, tiempo, hidden)
        pooled = out.mean(dim=1)
        return self.clasificador(pooled).squeeze(-1)  # logits
