"""
Transformer encoder pequeño entrenado desde cero para clasificación
binaria de pronunciación (correcto / incorrecto).

Es agnóstico a la representación de entrada: recibe una secuencia de
vectores de dimensión `input_dim`, ya sean frames del Mel-spectrograma
(input_dim = n_mels) o segmentos (input_dim = 2 * n_mels + 1).
"""

import torch
import torch.nn as nn

from src.config import AUDIO, MODEL


class PronunciationTransformer(nn.Module):
    def __init__(self, input_dim: int = AUDIO.n_mels, max_seq_len: int = MODEL.max_seq_len):
        super().__init__()
        self.proyeccion = nn.Linear(input_dim, MODEL.d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, MODEL.d_model) * 0.02
        )

        capa = nn.TransformerEncoderLayer(
            d_model=MODEL.d_model,
            nhead=MODEL.nhead,
            dim_feedforward=MODEL.dim_feedforward,
            dropout=MODEL.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa, num_layers=MODEL.num_layers)
        self.clasificador = nn.Linear(MODEL.d_model, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, longitud, input_dim)
        padding_mask: (batch, tiempo) -- True donde hay padding
        """
        seq_len = x.size(1)
        if seq_len > self.pos_encoding.size(1):
            raise ValueError(
                f"Secuencia de {seq_len} frames excede max_seq_len="
                f"{self.pos_encoding.size(1)}. Crea el modelo con un max_seq_len mayor."
            )

        x = self.proyeccion(x) + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.clasificador(x).squeeze(-1)  # logits