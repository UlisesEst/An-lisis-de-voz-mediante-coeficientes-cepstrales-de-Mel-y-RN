"""
Dataset y collate_fn con padding dinámico + máscara de atención.
"""

from pathlib import Path
from typing import Iterable, Union

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.config import AUDIO, SEGMENTATION, AudioConfig, SegmentationConfig
from src.preprocessing.mel_extraction import (
    compute_mel_db, load_and_extract, load_waveform, normalize_mel,
)
from src.preprocessing.segmentation import segment, segment_features

AudioSource = Union[str, Path, Iterable[Union[str, Path]], None]


def _resolve_class_files(source: AudioSource, default_dir: Path) -> list[Path]:
    """Resuelve una fuente de audios (carpeta, archivo único o lista de rutas)
    en una lista ordenada de archivos .wav."""
    if source is None:
        return sorted(default_dir.glob("*.wav")) if default_dir.exists() else []
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return sorted(p.glob("*.wav"))
        if p.is_file() and p.suffix.lower() == ".wav":
            return [p]
        return []
    return [Path(p) for p in source if Path(p).is_file()]


def resolve_audio_sources(
    root_dir: Union[str, Path] = "data/raw",
    correct_source: AudioSource = None,
    incorrect_source: AudioSource = None,
) -> tuple[list[Path], list[Path]]:
    """Devuelve las listas (correct_files, incorrect_files) permitiendo usar
    tanto una carpeta raíz tradicional (<root_dir>/correct y /incorrect) como
    dos fuentes independientes para la Clase 1 (correcto) y Clase 0 (incorrecto)."""
    root = Path(root_dir) if root_dir else Path("data/raw")
    correct_files = _resolve_class_files(correct_source, root / "correct")
    incorrect_files = _resolve_class_files(incorrect_source, root / "incorrect")
    return correct_files, incorrect_files


class YuhmuPronunciationDataset(Dataset):
    """
    Soporta tanto la estructura por carpeta raíz:
        <root_dir>/correct/*.wav
        <root_dir>/incorrect/*.wav
    como dos fuentes independientes (`correct_source` e `incorrect_source`),
    que pueden ser carpetas en cualquier ubicación o listas de archivos .wav.
    Etiqueta: 1 = pronunciación correcta (Clase Positiva), 0 = incorrecta (Clase Negativa).
    """

    def __init__(
        self,
        root_dir: str = "data/raw",
        transform=None,
        audio_cfg: AudioConfig = AUDIO,
        correct_source: AudioSource = None,
        incorrect_source: AudioSource = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path("data/raw")
        self.transform = transform
        self.audio_cfg = audio_cfg
        self.correct_files, self.incorrect_files = resolve_audio_sources(
            self.root_dir, correct_source, incorrect_source
        )
        self.samples = []

        for path in self.correct_files:
            self.samples.append((path, 1))
        for path in self.incorrect_files:
            self.samples.append((path, 0))

        if not self.samples:
            raise FileNotFoundError(
                "No se encontraron archivos .wav en las fuentes especificadas "
                f"(root_dir={self.root_dir}, correct_source={correct_source}, "
                f"incorrect_source={incorrect_source})."
            )

    def __len__(self):
        return len(self.samples)

    def get_item(self, idx: int, transform=None):
        path, label = self.samples[idx]
        mel = load_and_extract(str(path), self.audio_cfg)  # (tiempo, n_mels)
        active_transform = transform if transform is not None else self.transform
        if active_transform is not None:
            mel = active_transform(mel)
        return mel, torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.get_item(idx)

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

    def __init__(
        self,
        root_dir: str = "data/raw",
        audio_cfg: AudioConfig = AUDIO,
        seg_cfg: SegmentationConfig = SEGMENTATION,
        correct_source: AudioSource = None,
        incorrect_source: AudioSource = None,
        transform=None,
    ):
        super().__init__(
            root_dir,
            transform=transform,
            audio_cfg=audio_cfg,
            correct_source=correct_source,
            incorrect_source=incorrect_source,
        )
        self.seg_cfg = seg_cfg

    @property
    def feature_dim(self) -> int:
        return 2 * self.audio_cfg.n_mels + 1

    def get_item(self, idx: int, transform=None):
        path, label = self.samples[idx]
        mel_db = compute_mel_db(load_waveform(str(path), self.audio_cfg), self.audio_cfg)

        # La segmentación usa el espectrograma en dB sin normalizar; los
        # tokens se construyen con la misma normalización del modelo de frames.
        result = segment(mel_db.numpy(), self.audio_cfg, self.seg_cfg)
        frames_tensor = normalize_mel(mel_db)
        active_transform = transform if transform is not None else self.transform
        if active_transform is not None:
            frames_tensor = active_transform(frames_tensor)
        frames = frames_tensor.numpy()
        frame_ms = 1000 * self.audio_cfg.hop_length / self.audio_cfg.sample_rate
        tokens = segment_features(frames, result.segments, frame_ms)

        return torch.from_numpy(tokens).float(), torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def max_frames(self) -> int:
        """Cota superior del número de segmentos: la zona con voz dividida
        entre la duración mínima de segmento."""
        frames = super().max_frames()
        min_frames = max(1, int(self.seg_cfg.min_segment_ms / 1000
                                * self.audio_cfg.sample_rate / self.audio_cfg.hop_length))
        return frames // min_frames + 1


class WaveformPronunciationDataset(YuhmuPronunciationDataset):
    """Dataset de forma de onda cruda normalizada (media 0, varianza 1)
    para modelos preentrenados como Wav2Vec2Classifier."""

    def __init__(
        self,
        root_dir: str = "data/raw",
        audio_cfg: AudioConfig = AUDIO,
        correct_source: AudioSource = None,
        incorrect_source: AudioSource = None,
    ):
        super().__init__(
            root_dir,
            transform=None,
            audio_cfg=audio_cfg,
            correct_source=correct_source,
            incorrect_source=incorrect_source,
        )

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = load_waveform(str(path), self.audio_cfg).squeeze(0)  # (muestras,)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        return waveform, torch.tensor(label, dtype=torch.float32)

    def max_frames(self) -> int:
        """Número máximo de muestras de audio crudo en el dataset."""
        longest = max(sf.info(str(path)).duration for path, _ in self.samples)
        return int(longest * self.audio_cfg.sample_rate) + 1


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