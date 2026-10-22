"""
Diagnóstico para el modelo segmentado: ¿el número de segmentos por sí
solo predice la clase? Evalúa clasificadores triviales (regresión
logística) con validación cruzada, usando como única información:
  - el número de segmentos,
  - la duración (frames del Mel-spectrograma),
  - ambos.
Si alguno se acerca al desempeño del Transformer segmentado, el modelo
podría estar ganando por un atajo en lugar de por el contenido acústico.

Uso:
    python -m src.training.diagnose_segments
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AUDIO, SEGMENTATION, TRAIN
from src.datasets.yuhmu_dataset import YuhmuPronunciationDataset
from src.preprocessing.mel_extraction import compute_mel_db, load_waveform
from src.preprocessing.segmentation import segment


def main():
    dataset = YuhmuPronunciationDataset("data/raw")
    n_segments, n_frames, labels = [], [], []

    for path, label in dataset.samples:
        mel_db = compute_mel_db(load_waveform(str(path), AUDIO), AUDIO).numpy()
        result = segment(mel_db, AUDIO, SEGMENTATION)
        n_segments.append(len(result.segments))
        n_frames.append(mel_db.shape[1])
        labels.append(label)

    n_segments = np.array(n_segments)
    n_frames = np.array(n_frames)
    labels = np.array(labels)

    print("=== Número de segmentos por clase ===")
    for cls, nombre in [(1, "correcto"), (0, "incorrecto")]:
        subset = n_segments[labels == cls]
        print(f"{nombre}: media={subset.mean():.2f}  std={subset.std():.2f}  "
              f"min={subset.min()}  max={subset.max()}")
    corr = np.corrcoef(n_segments, n_frames)[0, 1]
    print(f"Correlación entre número de segmentos y duración: {corr:.2f}")

    skf = StratifiedKFold(n_splits=TRAIN.k_folds, shuffle=True, random_state=TRAIN.seed)
    variantes = {
        "Solo número de segmentos": n_segments.reshape(-1, 1),
        "Solo duración": n_frames.reshape(-1, 1),
        "Segmentos + duración": np.column_stack([n_segments, n_frames]),
    }

    print("\n=== Baselines triviales (accuracy, k-fold) ===")
    for nombre, X in variantes.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")
        print(f"{nombre:28s}: {scores.mean():.4f} (+/- {scores.std():.4f})")


if __name__ == "__main__":
    main()