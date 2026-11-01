"""
Diagnóstico: ¿el modelo aprende pronunciación real, o solo la duración
del audio? Compara estadísticas de duración por clase y evalúa un
clasificador trivial (regresión logística) usando SOLO el número de
frames como feature.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import AUDIO, TRAIN, AudioConfig
from src.datasets.yuhmu_dataset import YuhmuPronunciationDataset


def run_duration_diagnosis(
    root_dir: str = "data/raw",
    audio_cfg: AudioConfig = AUDIO,
    correct_source=None,
    incorrect_source=None,
    k_folds: int = TRAIN.k_folds,
    seed: int = TRAIN.seed,
) -> dict:
    """Evalúa un clasificador trivial basado solo en la duración (frames)
    y devuelve estadísticas por clase y accuracy k-fold."""
    dataset = YuhmuPronunciationDataset(
        root_dir=root_dir,
        audio_cfg=audio_cfg,
        correct_source=correct_source,
        incorrect_source=incorrect_source,
    )

    lengths = []
    labels = []
    for mel, label in dataset:
        lengths.append(mel.shape[0])
        labels.append(int(label.item()))

    lengths = np.array(lengths)
    labels = np.array(labels)

    stats = {}
    for cls, nombre in [(1, "correcto"), (0, "incorrecto")]:
        subset = lengths[labels == cls]
        stats[nombre] = {
            "mean": float(subset.mean()),
            "std": float(subset.std()),
            "min": int(subset.min()),
            "max": int(subset.max()),
        }

    X = lengths.reshape(-1, 1)
    clf = LogisticRegression()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")

    return {
        "stats_by_class": stats,
        "fold_scores": scores.tolist(),
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "lengths": lengths.tolist(),
        "labels": labels.tolist(),
    }


def main():
    res = run_duration_diagnosis()

    print("=== Duración (frames) por clase ===")
    for nombre, st in res["stats_by_class"].items():
        print(f"{nombre}: media={st['mean']:.1f}  std={st['std']:.1f}  "
              f"min={st['min']}  max={st['max']}")

    print("\n=== Baseline trivial (solo duración) ===")
    print(f"Accuracy por fold: {np.array(res['fold_scores'])}")
    print(f"Accuracy promedio: {res['mean_accuracy']:.4f} (+/- {res['std_accuracy']:.4f})")


if __name__ == "__main__":
    main()
