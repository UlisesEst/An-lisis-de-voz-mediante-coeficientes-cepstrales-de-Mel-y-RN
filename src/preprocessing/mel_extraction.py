"""
Extracción de Mel-spectrograma parametrizable.

Todas las funciones reciben un AudioConfig; si no se indica, usan los
valores por defecto de configuración acústica.
"""

import soundfile as sf
import torch
import torchaudio

from src.config import AUDIO, AudioConfig


def get_mel_transform(cfg: AudioConfig = AUDIO) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        window_fn=torch.hann_window,
    )


def load_waveform(path: str, cfg: AudioConfig = AUDIO) -> torch.Tensor:
    """Carga el audio en mono y lo lleva al sample rate configurado.
    Devuelve un tensor de forma (1, muestras)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # a mono

    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
    return waveform


def compute_mel_db(waveform: torch.Tensor, cfg: AudioConfig = AUDIO) -> torch.Tensor:
    """Mel-spectrograma en dB, SIN normalizar. Forma (n_mels, tiempo)."""
    mel = get_mel_transform(cfg)(waveform)
    return torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0)


def normalize_mel(mel_db: torch.Tensor) -> torch.Tensor:
    """Normalización por muestra (media 0, desviación 1) y transposición a
    (tiempo, n_mels). Compartida por el modelo de frames y el segmentado
    para que ambos vean exactamente la misma escala."""
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.transpose(0, 1)


def load_and_extract(path: str, cfg: AudioConfig = AUDIO) -> torch.Tensor:
    """Mel-spectrograma normalizado por muestra, con forma
    (tiempo, n_mels) -- lista para el Transformer."""
    return normalize_mel(compute_mel_db(load_waveform(path, cfg), cfg))