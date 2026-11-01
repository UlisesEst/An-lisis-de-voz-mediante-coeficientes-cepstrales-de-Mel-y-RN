"""
Script y funciones de entrenamiento: entrena clasificadores de audio (Transformer
por frames, Transformer por segmentos o Wav2Vec2) con validación cruzada k-fold,
reporta métricas promedio, guarda un registro JSON de la corrida en results/ y
opcionalmente exporta el paquete de modelo entrenado (.pt + .json) en models/.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import (
    AUDIO, MODEL, SEGMENTATION, TRAIN,
    AudioConfig, ModelConfig, SegmentationConfig, TrainConfig,
)
from src.datasets.yuhmu_dataset import (
    SegmentedPronunciationDataset, WaveformPronunciationDataset, YuhmuPronunciationDataset,
)
from src.models.checkpoint import save_model_package
from src.models.transformer_scratch import PronunciationTransformer
from src.models.wav2vec_finetune import Wav2Vec2Classifier
from src.training.cross_validation import run_kfold


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con k-fold")
    parser.add_argument(
        "--representation",
        choices=["frames", "segments", "wav2vec2"],
        default="frames",
        help="frames: un token por frame; segments: un token por segmento; wav2vec2: onda cruda",
    )
    parser.add_argument("--sample-rate", type=int, default=AUDIO.sample_rate)
    parser.add_argument("--n-mels", type=int, default=AUDIO.n_mels)
    parser.add_argument("--win-ms", type=float, default=AUDIO.win_length_ms)
    parser.add_argument("--overlap", type=float, default=AUDIO.overlap)
    parser.add_argument("--correct-dir", type=str, default=None, help="Carpeta de audios correctos")
    parser.add_argument("--incorrect-dir", type=str, default=None, help="Carpeta de audios incorrectos")
    parser.add_argument("--spec-augment", action="store_true", help="Activar SpecAugment en entrenamiento")
    parser.add_argument("--save-model", type=str, default=None, help="Ruta opcional para guardar el modelo .pt")
    args = parser.parse_args()
    audio_cfg = AudioConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        win_length_ms=args.win_ms,
        overlap=args.overlap,
    )
    train_cfg = TrainConfig(use_spec_augment=args.spec_augment)
    return audio_cfg, train_cfg, args.representation, args.correct_dir, args.incorrect_dir, args.save_model


def build_dataset(
    representation: str,
    audio_cfg: AudioConfig = AUDIO,
    seg_cfg: SegmentationConfig = SEGMENTATION,
    root_dir: str = "data/raw",
    correct_source=None,
    incorrect_source=None,
):
    """Dataset y dimensión de entrada del modelo para cada representación.
    Compartida por train.py, permutation_test.py y la interfaz de escritorio."""
    if representation == "segments":
        dataset = SegmentedPronunciationDataset(
            root_dir=root_dir,
            audio_cfg=audio_cfg,
            seg_cfg=seg_cfg,
            correct_source=correct_source,
            incorrect_source=incorrect_source,
        )
        return dataset, dataset.feature_dim
    if representation == "wav2vec2":
        dataset = WaveformPronunciationDataset(
            root_dir=root_dir,
            audio_cfg=audio_cfg,
            correct_source=correct_source,
            incorrect_source=incorrect_source,
        )
        return dataset, 1
    dataset = YuhmuPronunciationDataset(
        root_dir=root_dir,
        audio_cfg=audio_cfg,
        correct_source=correct_source,
        incorrect_source=incorrect_source,
    )
    return dataset, audio_cfg.n_mels


def _load_compatible_weights(model: torch.nn.Module, state_dict: dict):
    """Carga pesos preentrenados adaptando pos_encoding si max_seq_len cambió."""
    own_state = model.state_dict()
    adapted = {}
    for k, v in state_dict.items():
        if k not in own_state:
            continue
        if k == "pos_encoding" and v.shape != own_state[k].shape and v.shape[2] == own_state[k].shape[2]:
            new_pe = own_state[k].clone()
            min_len = min(v.shape[1], new_pe.shape[1])
            new_pe[:, :min_len, :] = v[:, :min_len, :]
            adapted[k] = new_pe
        elif v.shape == own_state[k].shape:
            adapted[k] = v
    own_state.update(adapted)
    model.load_state_dict(own_state)


def build_model_fn(
    representation: str,
    input_dim: int,
    max_len: int,
    model_cfg: ModelConfig = MODEL,
    freeze_wav2vec: bool = True,
    initial_state_dict: Optional[dict] = None,
):
    """Devuelve la función constructora del modelo según la representación.
    Si se proporciona `initial_state_dict`, inicializa cada fold con dichos pesos."""
    def _factory():
        if representation == "wav2vec2":
            m = Wav2Vec2Classifier(freeze_encoder=freeze_wav2vec)
        else:
            m = PronunciationTransformer(
                input_dim=input_dim,
                max_seq_len=max_len,
                model_cfg=model_cfg,
            )
        if initial_state_dict is not None:
            _load_compatible_weights(m, initial_state_dict)
        return m

    return _factory


def run_training_experiment(
    representation: str = "frames",
    audio_cfg: AudioConfig = AUDIO,
    seg_cfg: SegmentationConfig = SEGMENTATION,
    model_cfg: ModelConfig = MODEL,
    train_cfg: TrainConfig = TRAIN,
    root_dir: str = "data/raw",
    correct_source=None,
    incorrect_source=None,
    freeze_wav2vec: bool = True,
    initial_state_dict: Optional[dict] = None,
    save_model_path: Optional[str] = None,
    model_label: Optional[str] = None,
    class_names: tuple[str, str] = ("Incorrecto", "Correcto"),
    progress_callback=None,
) -> dict:
    """Ejecuta un experimento completo de validación cruzada, guarda el
    registro JSON en results/ y opcionalmente exporta el mejor modelo en models/."""
    dataset, input_dim = build_dataset(
        representation,
        audio_cfg=audio_cfg,
        seg_cfg=seg_cfg,
        root_dir=root_dir,
        correct_source=correct_source,
        incorrect_source=incorrect_source,
    )
    max_len = max(model_cfg.max_seq_len, dataset.max_frames())

    fold_results, best_model = run_kfold(
        dataset,
        model_fn=build_model_fn(
            representation,
            input_dim,
            max_len,
            model_cfg=model_cfg,
            freeze_wav2vec=freeze_wav2vec,
            initial_state_dict=initial_state_dict,
        ),
        train_cfg=train_cfg,
        progress_callback=progress_callback,
        return_best_model=True,
    )

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        values = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"run_{representation}_{timestamp}.json"

    effective_model_cfg = ModelConfig(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        max_seq_len=max_len,
    )

    saved_pt = None
    if save_model_path and best_model is not None:
        saved_pt = save_model_package(
            model=best_model,
            save_path=save_model_path,
            representation=representation,
            input_dim=input_dim,
            max_seq_len=max_len,
            audio_cfg=audio_cfg,
            seg_cfg=seg_cfg if representation == "segments" else None,
            model_cfg=effective_model_cfg,
            train_cfg=train_cfg,
            metrics_summary=summary,
            class_names=class_names,
            model_label=model_label,
        )

    record = {
        "representation": representation,
        "audio_config": audio_cfg.to_dict(),
        "segmentation_config": asdict(seg_cfg) if representation == "segments" else None,
        "model_config": asdict(effective_model_cfg),
        "train_config": asdict(train_cfg),
        "folds": fold_results,
        "summary": summary,
        "saved_model_path": str(saved_pt) if saved_pt else None,
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "record": record,
        "result_json_path": out_path,
        "saved_model_path": saved_pt,
        "best_model": best_model,
        "input_dim": input_dim,
        "max_seq_len": max_len,
    }


def main():
    audio_cfg, train_cfg, representation, correct_dir, incorrect_dir, save_model = parse_args()
    for w in audio_cfg.validate():
        print(f"Advertencia: {w}")

    res = run_training_experiment(
        representation=representation,
        audio_cfg=audio_cfg,
        train_cfg=train_cfg,
        correct_source=correct_dir,
        incorrect_source=incorrect_dir,
        save_model_path=save_model,
    )

    print("\n=== Resultados promedio (k-fold) ===")
    for key, val in res["record"]["summary"].items():
        print(f"{key}: {val['mean']:.4f} (+/- {val['std']:.4f})")
    print(f"\nRegistro guardado en {res['result_json_path']}")
    if res["saved_model_path"]:
        print(f"Modelo guardado en {res['saved_model_path']}")


if __name__ == "__main__":
    main()
