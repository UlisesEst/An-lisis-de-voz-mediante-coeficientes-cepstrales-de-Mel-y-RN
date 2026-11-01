"""
Motor de inferencia: evalúa uno o varios archivos .wav utilizando un modelo
entrenado (.pt) y aplica automáticamente la configuración exacta de audio
y segmentación con la que dicho modelo fue entrenado.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from src.config import AudioConfig, SegmentationConfig
from src.preprocessing.mel_extraction import compute_mel_db, load_waveform, normalize_mel
from src.preprocessing.segmentation import quality_indicators, segment, segment_features


def predict_audio(
    audio_path: Union[str, Path],
    model: nn.Module,
    metadata: dict,
    device: Optional[str] = None,
) -> dict:
    """Evalúa un archivo .wav individual y devuelve la predicción binaria,
    probabilidades y representaciones espectrales para visualización."""
    device = device or next(model.parameters()).device
    audio_path = Path(audio_path)

    audio_cfg = AudioConfig.from_dict(metadata.get("audio_config") or {})
    seg_cfg = SegmentationConfig.from_dict(metadata.get("segmentation_config") or {})
    representation = metadata.get("representation", "frames")
    max_seq_len = int(metadata.get("max_seq_len", 500))
    class_names = metadata.get("class_names") or {"0": "Incorrecto", "1": "Correcto"}

    waveform_tensor = load_waveform(str(audio_path), audio_cfg)  # (1, muestras)
    waveform_1d = waveform_tensor.squeeze(0)
    mel_db = compute_mel_db(waveform_tensor, audio_cfg)  # (n_mels, tiempo)
    mel_db_np = mel_db.numpy()
    seg_result = segment(mel_db_np, audio_cfg, seg_cfg)
    ind = quality_indicators(seg_result, audio_cfg)

    truncated = False
    if representation == "frames":
        features = normalize_mel(mel_db)  # (tiempo, n_mels)
        if features.shape[0] > max_seq_len:
            features = features[:max_seq_len]
            truncated = True
        x = features.unsqueeze(0).to(device)  # (1, tiempo, n_mels)
    elif representation == "segments":
        frames_norm = normalize_mel(mel_db).numpy()
        frame_ms = 1000 * audio_cfg.hop_length / audio_cfg.sample_rate
        tokens = segment_features(frames_norm, seg_result.segments, frame_ms)
        if tokens.shape[0] > max_seq_len:
            tokens = tokens[:max_seq_len]
            truncated = True
        x = torch.from_numpy(tokens).float().unsqueeze(0).to(device)
    elif representation == "wav2vec2":
        w_norm = (waveform_1d - waveform_1d.mean()) / (waveform_1d.std() + 1e-6)
        x = w_norm.unsqueeze(0).to(device)
    else:
        raise ValueError(f"Representación no soportada: {representation}")

    model.eval()
    with torch.no_grad():
        logit = model(x)
        prob_class_1 = float(torch.sigmoid(logit).item())

    pred_label = 1 if prob_class_1 >= 0.5 else 0
    confidence = prob_class_1 if pred_label == 1 else (1.0 - prob_class_1)
    pred_name = class_names.get(str(pred_label), "Correcto" if pred_label == 1 else "Incorrecto")

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path),
        "pred_label": pred_label,
        "pred_class_name": pred_name,
        "prob_correct": prob_class_1,
        "prob_incorrect": 1.0 - prob_class_1,
        "confidence": confidence,
        "duration_s": float(len(waveform_1d) / audio_cfg.sample_rate),
        "n_frames": int(mel_db_np.shape[1]),
        "n_segments": int(len(seg_result.segments)),
        "truncated": truncated,
        "waveform": waveform_1d.numpy(),
        "sample_rate": audio_cfg.sample_rate,
        "mel_db": mel_db_np,
        "audio_cfg": audio_cfg,
        "seg_cfg": seg_cfg,
        "segmentation": seg_result,
        "quality": ind,
    }


def predict_batch(
    audio_paths: Iterable[Union[str, Path]],
    model: nn.Module,
    metadata: dict,
    device: Optional[str] = None,
) -> list[dict]:
    """Evalúa un lote de archivos .wav y devuelve la lista de resultados."""
    return [predict_audio(p, model, metadata, device=device) for p in audio_paths]
