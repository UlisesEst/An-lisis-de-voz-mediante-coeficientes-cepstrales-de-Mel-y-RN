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

from src.config import MODEL, TRAIN
from src.models.transformer_scratch import PronunciationTransformer
from src.training.cross_validation import run_kfold
from src.training.train import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Test de permutación de etiquetas")
    parser.add_argument("--representation", choices=["frames", "segments"], default="frames")
    representation = parser.parse_args().representation

    dataset, input_dim = build_dataset(representation)
    max_len = max(MODEL.max_seq_len, dataset.max_frames())

    labels = [label for _, label in dataset.samples]
    random.seed(TRAIN.seed)
    random.shuffle(labels)
    dataset.samples = [(path, labels[i]) for i, (path, _) in enumerate(dataset.samples)]

    print(f"Representación: {representation}. Etiquetas mezcladas al azar. "
          "Entrenando (debería dar ~0.50 si no hay fuga)...")
    fold_results = run_kfold(
        dataset,
        model_fn=lambda: PronunciationTransformer(input_dim=input_dim, max_seq_len=max_len),
    )

    print(f"\n=== Resultados con etiquetas ALEATORIAS ({representation}) ===")
    for key in ["accuracy", "f1"]:
        values = [r[key] for r in fold_results]
        print(f"{key}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")


if __name__ == "__main__":
    main()