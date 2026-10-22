"""
Diagnóstico: ¿el modelo aprende pronunciación real, o solo la duración
del audio? Compara estadísticas de duración por clase y evalúa un
clasificador trivial (regresión logística) usando SOLO el número de
frames como feature.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.datasets.yuhmu_dataset import YuhmuPronunciationDataset


def main():
    dataset = YuhmuPronunciationDataset("data/raw")

    lengths = []
    labels = []
    for mel, label in dataset:
        lengths.append(mel.shape[0])
        labels.append(int(label.item()))

    lengths = np.array(lengths)
    labels = np.array(labels)

    print("=== Duración (frames) por clase ===")
    for cls, nombre in [(1, "correcto"), (0, "incorrecto")]:
        subset = lengths[labels == cls]
        print(f"{nombre}: media={subset.mean():.1f}  std={subset.std():.1f}  "
              f"min={subset.min()}  max={subset.max()}")

    X = lengths.reshape(-1, 1)
    clf = LogisticRegression()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")

    print("\n=== Baseline trivial (solo duración) ===")
    print(f"Accuracy por fold: {scores}")
    print(f"Accuracy promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")


if __name__ == "__main__":
    main()