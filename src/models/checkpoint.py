"""
Gestión de guardado y carga de modelos entrenados (.pt + metadatos JSON).

Cada paquete guardado conserva:
  - Los pesos entrenados del modelo (`state_dict`).
  - La arquitectura/representación (`frames`, `segments` o `wav2vec2`).
  - La configuración exacta de audio (`AudioConfig`), segmentación
    (`SegmentationConfig`) e hiperparámetros (`ModelConfig`, `TrainConfig`)
    necesarios para reproducir el preprocesamiento durante la inferencia.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from src.config import (
    AUDIO, MODEL, SEGMENTATION, TRAIN,
    AudioConfig, ModelConfig, SegmentationConfig, TrainConfig,
)
from src.models.transformer_scratch import PronunciationTransformer
from src.models.wav2vec_finetune import MODEL_NAME as DEFAULT_WAV2VEC_NAME, Wav2Vec2Classifier


def save_model_package(
    model: nn.Module,
    save_path: Union[str, Path],
    representation: str,
    input_dim: int,
    max_seq_len: int,
    audio_cfg: AudioConfig = AUDIO,
    seg_cfg: Optional[SegmentationConfig] = SEGMENTATION,
    model_cfg: ModelConfig = MODEL,
    train_cfg: Optional[TrainConfig] = TRAIN,
    metrics_summary: Optional[dict] = None,
    class_names: tuple[str, str] = ("Incorrecto", "Correcto"),
    model_label: Optional[str] = None,
) -> Path:
    """Guarda el modelo entrenado y toda su configuración en un archivo .pt
    junto con un archivo .json legible."""
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".pt":
        save_path = save_path.with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().isoformat(timespec="seconds")
    label = model_label or save_path.stem

    metadata = {
        "model_label": label,
        "representation": representation,
        "input_dim": int(input_dim),
        "max_seq_len": int(max_seq_len),
        "audio_config": audio_cfg.to_dict(),
        "segmentation_config": seg_cfg.to_dict() if seg_cfg is not None else None,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict() if train_cfg is not None else None,
        "class_names": {"0": class_names[0], "1": class_names[1]},
        "summary": metrics_summary or {},
        "created_at": created_at,
    }

    if representation == "wav2vec2" and isinstance(model, Wav2Vec2Classifier):
        metadata["freeze_encoder"] = bool(getattr(model, "freeze_encoder", True))
        metadata["wav2vec_model_name"] = getattr(model, "model_name", DEFAULT_WAV2VEC_NAME)
        if metadata["freeze_encoder"]:
            # Si el encoder de Wav2Vec2 estuvo congelado, solo guardamos la capa
            # clasificadora para que el archivo .pt pese unos pocos KB en vez de ~360 MB.
            state_to_save = {"classifier_only": True, "state_dict": model.clasificador.state_dict()}
        else:
            state_to_save = {"classifier_only": False, "state_dict": model.state_dict()}
    else:
        state_to_save = {"classifier_only": False, "state_dict": model.state_dict()}

    payload = {
        "metadata": metadata,
        **state_to_save,
    }
    torch.save(payload, save_path)

    json_path = save_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return save_path


def load_model_package(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> tuple[nn.Module, dict]:
    """Carga un archivo .pt guardado, reconstruye el modelo con sus dimensiones
    y configuración originales, y lo devuelve en modo evaluación junto con sus metadatos."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "metadata" not in payload:
        raise ValueError(
            f"El archivo '{checkpoint_path.name}' no contiene metadatos compatibles con esta herramienta."
        )

    meta = payload["metadata"]
    representation = meta["representation"]
    input_dim = int(meta["input_dim"])
    max_seq_len = int(meta["max_seq_len"])
    model_cfg = ModelConfig.from_dict(meta.get("model_config") or {})

    if representation == "wav2vec2":
        freeze_enc = bool(meta.get("freeze_encoder", True))
        w2v_name = meta.get("wav2vec_model_name", DEFAULT_WAV2VEC_NAME)
        model = Wav2Vec2Classifier(freeze_encoder=freeze_enc, model_name=w2v_name)
        if payload.get("classifier_only", False):
            model.clasificador.load_state_dict(payload["state_dict"])
        else:
            model.load_state_dict(payload["state_dict"])
    else:
        model = PronunciationTransformer(
            input_dim=input_dim,
            max_seq_len=max_seq_len,
            model_cfg=model_cfg,
        )
        model.load_state_dict(payload["state_dict"])

    model = model.to(device)
    model.eval()
    return model, meta


def list_saved_models(models_dir: Union[str, Path] = "models") -> list[dict]:
    """Lista todos los modelos .pt disponibles en la carpeta `models/` junto
    con sus metadatos básicos."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    items = []
    for pt_file in sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        json_file = pt_file.with_suffix(".json")
        meta = {}
        if json_file.exists():
            try:
                meta = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        items.append({
            "path": pt_file,
            "name": pt_file.name,
            "label": meta.get("model_label", pt_file.stem),
            "representation": meta.get("representation", "desconocido"),
            "created_at": meta.get("created_at", ""),
            "summary": meta.get("summary", {}),
            "metadata": meta,
        })
    return items
