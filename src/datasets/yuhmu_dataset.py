"""
Dataset y collate_fn con padding dinámico + máscara de atención.
"""

from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.config import AUDIO, SEGMENTATION, AudioConfig, SegmentationConfig
from src.preprocessing.mel_extraction import (
    compute_mel_db, load_and_extract, load_waveform, normalize_mel,
)
from src.preprocessing.segmentation import segment, segment_features


class YuhmuPronunciationDataset(Dataset):
    """
    Espera la estructura:
        <root_dir>/correct/*.wav
        <root_dir>/incorrect/*.wav
    Etiqueta: 1 = pronunciación correcta, 0 = incorrecta.
    """

    def __init__(self, root_dir: str, transform=None, audio_cfg: AudioConfig = AUDIO):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.audio_cfg = audio_cfg
        self.samples = []

        for path in sorted((self.root_dir / "correct").glob("*.wav")):
            self.samples.append((path, 1))
        for path in sorted((self.root_dir / "incorrect").glob("*.wav")):
            self.samples.append((path, 0))

        if not self.samples:
            raise FileNotFoundError(
                f"No se encontraron .wav en {self.root_dir}/correct o /incorrect"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = load_and_extract(str(path), self.audio_cfg)  # (tiempo, n_mels)
        if self.transform is not None:
            mel = self.transform(mel)
        return mel, torch.tensor(label, dtype=torch.float32)

    def max_frames(self) -> int:
        """Número máximo de frames que producirá el audio más largo, calculado
        a partir de los metadatos (sin decodificar los archivos)."""
        longest = max(sf.info(str(path)).duration for path, _ in self.samples)
        samples = int(longest * self.audio_cfg.sample_rate)
        return 1 + samples // self.audio_cfg.hop_length


class SegmentedPronunciationDataset(YuhmuPronunciationDataset):
    """Misma lista de archivos y etiquetas que YuhmuPronunciationDataset,
    pero cada audio se representa como una secuencia de segmentos
    (aprox. unidades fonéticas) en lugar de frames. Cada token contiene
    media, desviación estándar y log-duración del segmento."""

    def __init__(self, root_dir: str, audio_cfg: AudioConfig = AUDIO,
                 seg_cfg: SegmentationConfig = SEGMENTATION):
        super().__init__(root_dir, transform=None, audio_cfg=audio_cfg)
        self.seg_cfg = seg_cfg

    @property
    def feature_dim(self) -> int:
        return 2 * self.audio_cfg.n_mels + 1

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel_db = compute_mel_db(load_waveform(str(path), self.audio_cfg), self.audio_cfg)

        # La segmentación usa el espectrograma en dB sin normalizar; los
        # tokens se construyen con la misma normalización del modelo de frames.
        result = segment(mel_db.numpy(), self.audio_cfg, self.seg_cfg)
        frames = normalize_mel(mel_db).numpy()
        frame_ms = 1000 * self.audio_cfg.hop_length / self.audio_cfg.sample_rate
        tokens = segment_features(frames, result.segments, frame_ms)

        return torch.from_numpy(tokens).float(), torch.tensor(label, dtype=torch.float32)

    def max_frames(self) -> int:
        """Cota superior del número de segmentos: la zona con voz dividida
        entre la duración mínima de segmento."""
        frames = super().max_frames()
        min_frames = max(1, int(self.seg_cfg.min_segment_ms / 1000
                                * self.audio_cfg.sample_rate / self.audio_cfg.hop_length))
        return frames // min_frames + 1


def collate_fn(batch):
    """
    Aplica padding a la dimensión temporal y genera la máscara
    (True = posición de padding, para src_key_padding_mask).
    """
    mels, labels = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mels])

    padded = pad_sequence(mels, batch_first=True)  # (batch, max_tiempo, n_mels)

    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # (batch, max_tiempo)

    labels = torch.stack(labels)
    return padded, mask, labels