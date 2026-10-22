"""
Segmentación espectral no supervisada por distancia coseno.

Detecta fronteras entre unidades fonéticas comparando frames consecutivos
del Mel-spectrograma: dentro de un fonema el espectro es estable y en la
frontera cambia bruscamente. Inspirado en el método de Ramos-Aguilar
et al. (2025), generalizado para cualquier idioma mediante umbrales
relativos a cada audio (sin calibración por idioma ni transcripciones).

Trabaja solo con NumPy/SciPy para poder reutilizarse fuera de PyTorch.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from src.config import AUDIO, SEGMENTATION, AudioConfig, SegmentationConfig

# Rango aproximado de duración de un fonema en la mayoría de los idiomas.
PLAUSIBLE_RANGE_MS = (30.0, 200.0)


@dataclass
class SegmentationResult:
    boundaries: np.ndarray  # frames donde inicia un nuevo segmento (fronteras internas)
    segments: list          # [(inicio, fin)] con fin exclusivo, solo zona con voz
    novelty: np.ndarray     # curva suavizada; novelty[t] = distancia entre frames t-1 y t
    threshold: float        # umbral adaptativo usado para esta grabación
    voiced_start: int       # primer frame con voz
    voiced_end: int         # frame posterior al último con voz (exclusivo)


def ms_to_frames(ms: float, audio_cfg: AudioConfig) -> int:
    return max(1, int(round(ms / 1000 * audio_cfg.sample_rate / audio_cfg.hop_length)))


def cosine_novelty(features: np.ndarray) -> np.ndarray:
    """Distancia coseno entre cada frame y el anterior.
    features: (tiempo, dimensiones). Devuelve un vector de longitud tiempo."""
    a, b = features[:-1], features[1:]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return np.concatenate([[0.0], 1.0 - num / den])


def window_novelty(features: np.ndarray, w: int) -> np.ndarray:
    """Distancia coseno entre el promedio de los `w` frames anteriores a t
    y el promedio de los `w` frames a partir de t (idea de la medida de
    novedad de Foote). A diferencia de comparar frames sueltos, detecta
    transiciones graduales (p. ej. vocal-vocal o vocal-semivocal) y el
    promedio atenúa el ruido. Con w = 1 equivale a cosine_novelty."""
    if w <= 1:
        return cosine_novelty(features)

    n = len(features)
    cumsum = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, axis=0)])
    t = np.arange(1, n)
    left_start = np.maximum(0, t - w)
    right_end = np.minimum(n, t + w)
    left = (cumsum[t] - cumsum[left_start]) / (t - left_start)[:, None]
    right = (cumsum[right_end] - cumsum[t]) / (right_end - t)[:, None]

    num = (left * right).sum(axis=1)
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-8
    return np.concatenate([[0.0], 1.0 - num / den])


def voiced_region(mel_db: np.ndarray, silence_db: float) -> tuple:
    """Recorta el silencio inicial y final: un frame es silencio si su
    energía media está más de `silence_db` dB por debajo del pico."""
    energy = mel_db.mean(axis=0)
    voiced = np.flatnonzero(energy >= energy.max() - silence_db)
    return int(voiced[0]), int(voiced[-1]) + 1


def segment(mel_db: np.ndarray, audio_cfg: AudioConfig = AUDIO,
            seg_cfg: SegmentationConfig = SEGMENTATION) -> SegmentationResult:
    """mel_db: Mel-spectrograma en dB SIN normalizar, forma (n_mels, tiempo)."""
    # 1. Normalización por banda: resalta cambios de forma espectral
    features = (mel_db - mel_db.mean(axis=1, keepdims=True)).T  # (tiempo, n_mels)

    # 2. Curva de novedad por ventanas, opcionalmente suavizada
    context = ms_to_frames(seg_cfg.context_ms, audio_cfg)
    novelty = window_novelty(features, context)
    width = ms_to_frames(seg_cfg.smoothing_ms, audio_cfg)
    if width > 1:
        novelty = np.convolve(novelty, np.ones(width) / width, mode="same")

    # 3. Zona con voz
    start, end = voiced_region(mel_db, seg_cfg.silence_db)
    min_frames = ms_to_frames(seg_cfg.min_segment_ms, audio_cfg)

    # 4. Umbral adaptativo calculado solo sobre la zona con voz. Se excluye
    #    un margen en cada borde: ahí las ventanas de contexto y el suavizado
    #    alcanzan el silencio y arrastran el pico de la transición
    #    silencio-voz, que inflaría el umbral.
    margin = context + width // 2 + 1
    region = novelty[start + margin:end - margin]
    if len(region) < 3:
        return SegmentationResult(np.array([], dtype=int), [(start, end)], novelty, 0.0, start, end)

    # Umbral relativo robusto: mediana + k * MAD. Con media y desviación
    # estándar, las fronteras fuertes inflaban el umbral y ocultaban las
    # moderadas; la mediana y la MAD describen el fondo típico de la curva
    # sin que los picos las arrastren. El factor 1.4826 hace la MAD
    # comparable a la desviación estándar; k = 3 es la regla habitual para
    # detectar valores atípicos, que es lo que una frontera representa.
    median = np.median(region)
    mad = 1.4826 * np.median(np.abs(region - median))
    relative = median + seg_cfg.threshold_k * mad

    # Aun así, el umbral relativo siempre deja picos sobre él en ruido puro;
    # el piso absoluto exige que el cambio sea real. La distancia coseno es
    # insensible al volumen y está acotada, así que su escala es comparable
    # entre grabaciones e idiomas.
    threshold = float(max(relative, seg_cfg.min_novelty))
    peaks, _ = find_peaks(region, height=threshold, distance=min_frames)
    boundaries = peaks + start + margin

    # Los segmentos de los extremos también deben respetar la duración mínima
    boundaries = boundaries[(boundaries - start >= min_frames) & (end - boundaries >= min_frames)]

    edges = [start, *boundaries.tolist(), end]
    segments = list(zip(edges[:-1], edges[1:]))
    return SegmentationResult(boundaries, segments, novelty, threshold, start, end)


def pool_segments(features: np.ndarray, segments: list) -> np.ndarray:
    """Resume cada segmento en un vector (promedio de sus frames).
    features: (tiempo, dimensiones). Devuelve (n_segmentos, dimensiones)."""
    return np.stack([features[s:e].mean(axis=0) for s, e in segments])


def segment_features(features: np.ndarray, segments: list, frame_ms: float) -> np.ndarray:
    """Un vector por segmento con tres componentes:
      - media de los frames: qué sonido es (color espectral típico)
      - desviación estándar: cuánto cambia dentro del segmento; compensa en
        parte la sub-segmentación (p. ej. transiciones vocálicas graduales)
      - log-duración relativa a 100 ms: la duración se pierde al promediar
        y puede ser distintiva (vocales largas/cortas, tono)
    features: (tiempo, D). Devuelve (n_segmentos, 2*D + 1)."""
    means = pool_segments(features, segments)
    stds = np.stack([features[s:e].std(axis=0) for s, e in segments])
    log_dur = np.array([[np.log((e - s) * frame_ms / 100.0)] for s, e in segments])
    return np.concatenate([means, stds, log_dur], axis=1)


def segment_durations_ms(result: SegmentationResult, audio_cfg: AudioConfig = AUDIO) -> np.ndarray:
    frame_ms = 1000 * audio_cfg.hop_length / audio_cfg.sample_rate
    return np.array([(e - s) * frame_ms for s, e in result.segments])


def quality_indicators(result: SegmentationResult, audio_cfg: AudioConfig = AUDIO) -> dict:
    """Indicadores de calidad que no requieren transcripción."""
    durations = segment_durations_ms(result, audio_cfg)
    low, high = PLAUSIBLE_RANGE_MS
    return {
        "n_segmentos": len(result.segments),
        "duracion_media_ms": float(durations.mean()),
        "pct_duracion_plausible": float(((durations >= low) & (durations <= high)).mean() * 100),
    }