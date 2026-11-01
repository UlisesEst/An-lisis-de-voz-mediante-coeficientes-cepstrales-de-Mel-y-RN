"""
Test de permutación: mezcla las etiquetas al azar y reentrena.
Si el modelo sigue logrando accuracy alta, hay fuga de datos en el
pipeline. Si cae a ~50% (azar), el resultado original es genuino.

Uso:
    python -m src.training.permutation_test                            # frames
    python -m src.training.permutation_test --representation segments  # segmentos
"""

import argparse
import random

import numpy as np

from src.config import AUDIO, MODEL, SEGMENTATION, TRAIN, AudioConfig, ModelConfig, SegmentationConfig, TrainConfig
from src.training.cross_validation import run_kfold
from src.training.train import build_dataset, build_model_fn


def execute_permutation_test(
    representation: str = "frames",
    audio_cfg: AudioConfig = AUDIO,
    seg_cfg: SegmentationConfig = SEGMENTATION,
    model_cfg: ModelConfig = MODEL,
    train_cfg: TrainConfig = TRAIN,
    root_dir: str = "data/raw",
    correct_source=None,
    incorrect_source=None,
    progress_callback=None,
) -> dict:
    """Mezcla las etiquetas al azar y entrena con k-fold para verificar que
    el accuracy caiga a ~0.50 (ausencia de fuga de datos)."""
    dataset, input_dim = build_dataset(
        representation,
        audio_cfg=audio_cfg,
        seg_cfg=seg_cfg,
        root_dir=root_dir,
        correct_source=correct_source,
        incorrect_source=incorrect_source,
    )
    max_len = max(model_cfg.max_seq_len, dataset.max_frames())

    labels = [label for _, label in dataset.samples]
    rng = random.Random(train_cfg.seed)
    rng.shuffle(labels)
    dataset.samples = [(path, labels[i]) for i, (path, _) in enumerate(dataset.samples)]

    fold_results = run_kfold(
        dataset,
        model_fn=build_model_fn(representation, input_dim, max_len, model_cfg),
        train_cfg=train_cfg,
        progress_callback=progress_callback,
    )

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        values = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {"folds": fold_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Test de permutación de etiquetas")
    parser.add_argument("--representation", choices=["frames", "segments", "wav2vec2"], default="frames")
    representation = parser.parse_args().representation

    print(f"Representación: {representation}. Etiquetas mezcladas al azar. "
          "Entrenando (debería dar ~0.50 si no hay fuga)...")
    res = execute_permutation_test(representation=representation)

    print(f"\n=== Resultados con etiquetas ALEATORIAS ({representation}) ===")
    for key in ["accuracy", "f1"]:
        st = res["summary"][key]
        print(f"{key}: {st['mean']:.4f} (+/- {st['std']:.4f})")


if __name__ == "__main__":
    main()
