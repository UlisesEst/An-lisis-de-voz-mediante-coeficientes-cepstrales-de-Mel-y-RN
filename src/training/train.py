"""
Script principal: entrena el Transformer desde cero con validación
cruzada k-fold, reporta métricas promedio y guarda un registro JSON de
la corrida (parámetros + resultados) en results/.

Uso (valores por defecto, validados para Yuhmu):
    python -m src.training.train

Con parámetros propios:
    python -m src.training.train --sample-rate 16000 --n-mels 64 --win-ms 25 --overlap 0.5

Con representación por segmentos (un token por unidad fonética):
    python -m src.training.train --representation segments
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import AUDIO, MODEL, SEGMENTATION, TRAIN, AudioConfig
from src.datasets.yuhmu_dataset import SegmentedPronunciationDataset, YuhmuPronunciationDataset
from src.models.transformer_scratch import PronunciationTransformer
from src.training.cross_validation import run_kfold


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con k-fold")
    parser.add_argument("--representation", choices=["frames", "segments"], default="frames",
                        help="frames: un token por frame; segments: un token por segmento")
    parser.add_argument("--sample-rate", type=int, default=AUDIO.sample_rate)
    parser.add_argument("--n-mels", type=int, default=AUDIO.n_mels)
    parser.add_argument("--win-ms", type=float, default=AUDIO.win_length_ms)
    parser.add_argument("--overlap", type=float, default=AUDIO.overlap)
    args = parser.parse_args()
    audio_cfg = AudioConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        win_length_ms=args.win_ms,
        overlap=args.overlap,
    )
    return audio_cfg, args.representation


def build_dataset(representation: str, audio_cfg: AudioConfig = AUDIO):
    """Dataset y dimensión de entrada del modelo para cada representación.
    Compartida por train.py y permutation_test.py para que ambos construyan
    los datos exactamente igual."""
    if representation == "segments":
        dataset = SegmentedPronunciationDataset("data/raw", audio_cfg=audio_cfg)
        return dataset, dataset.feature_dim
    dataset = YuhmuPronunciationDataset("data/raw", audio_cfg=audio_cfg)
    return dataset, audio_cfg.n_mels


def main():
    audio_cfg, representation = parse_args()
    for w in audio_cfg.validate():
        print(f"Advertencia: {w}")

    dataset, input_dim = build_dataset(representation, audio_cfg)
    max_len = max(MODEL.max_seq_len, dataset.max_frames())
    print(f"Dataset cargado: {len(dataset)} muestras | representación: {representation} "
          f"| dimensión de entrada: {input_dim} | parámetros: {audio_cfg.to_dict()}")

    fold_results = run_kfold(
        dataset,
        model_fn=lambda: PronunciationTransformer(input_dim=input_dim, max_seq_len=max_len),
    )

    summary = {}
    print("\n=== Resultados promedio (k-fold) ===")
    for key in ["accuracy", "precision", "recall", "f1"]:
        values = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        print(f"{key}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"run_{representation}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps({
        "representation": representation,
        "audio_config": audio_cfg.to_dict(),
        "segmentation_config": asdict(SEGMENTATION) if representation == "segments" else None,
        "model_config": asdict(MODEL) | {"max_seq_len": max_len},
        "train_config": asdict(TRAIN),
        "folds": fold_results,
        "summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRegistro guardado en {out_path}")


if __name__ == "__main__":
    main()