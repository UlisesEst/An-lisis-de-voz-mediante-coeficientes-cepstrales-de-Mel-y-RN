"""
Fine-tuning de un Wav2Vec2 preentrenado como comparación contra el
Transformer entrenado desde cero.

Nota: este modelo trabaja sobre la forma de onda cruda, NO sobre el
Mel-spectrograma -- Wav2Vec2 aprende su propia representación interna.
Utiliza WaveformPronunciationDataset y collate_fn de src/datasets/yuhmu_dataset.py.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, freeze_encoder: bool = True, model_name: str = MODEL_NAME):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.model_name = model_name
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.clasificador = nn.Linear(hidden_size, 1)

    def forward(self, waveform: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        waveform: (batch, samples)
        padding_mask: (batch, samples) -- True donde hay padding (compatible con collate_fn)
        """
        if waveform.dim() == 3 and waveform.size(-1) == 1:
            waveform = waveform.squeeze(-1)

        attention_mask = (~padding_mask).long() if padding_mask is not None else None
        if attention_mask is not None and getattr(self.encoder.config, "feat_extract_norm", "group") == "layer":
            out = self.encoder(waveform, attention_mask=attention_mask).last_hidden_state
        else:
            out = self.encoder(waveform).last_hidden_state  # (batch, tiempo, hidden)

        if attention_mask is not None and hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            feat_mask = self.encoder._get_feature_vector_attention_mask(
                out.shape[1], attention_mask
            ).unsqueeze(-1).float()
            pooled = (out * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        return self.clasificador(pooled).squeeze(-1)  # logits
