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

from src.config import AUDIO, SEGMENTATION, TRAIN, AudioConfig, SegmentationConfig
from src.datasets.yuhmu_dataset import YuhmuPronunciationDataset
from src.preprocessing.mel_extraction import compute_mel_db, load_waveform
from src.preprocessing.segmentation import segment


def run_segments_diagnosis(
    root_dir: str = "data/raw",
    audio_cfg: AudioConfig = AUDIO,
    seg_cfg: SegmentationConfig = SEGMENTATION,
    correct_source=None,
    incorrect_source=None,
    k_folds: int = TRAIN.k_folds,
    seed: int = TRAIN.seed,
) -> dict:
    """Evalúa clasificadores triviales basados en número de segmentos y/o duración."""
    dataset = YuhmuPronunciationDataset(
        root_dir=root_dir,
        audio_cfg=audio_cfg,
        correct_source=correct_source,
        incorrect_source=incorrect_source,
    )
    n_segments, n_frames, labels = [], [], []

    for path, label in dataset.samples:
        mel_db = compute_mel_db(load_waveform(str(path), audio_cfg), audio_cfg).numpy()
        result = segment(mel_db, audio_cfg, seg_cfg)
        n_segments.append(len(result.segments))
        n_frames.append(mel_db.shape[1])
        labels.append(label)

    n_segments = np.array(n_segments)
    n_frames = np.array(n_frames)
    labels = np.array(labels)

    stats = {}
    for cls, nombre in [(1, "correcto"), (0, "incorrecto")]:
        subset = n_segments[labels == cls]
        stats[nombre] = {
            "mean": float(subset.mean()),
            "std": float(subset.std()),
            "min": int(subset.min()),
            "max": int(subset.max()),
        }
    corr = float(np.corrcoef(n_segments, n_frames)[0, 1]) if len(n_segments) > 1 else 0.0

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    variantes = {
        "Solo número de segmentos": n_segments.reshape(-1, 1),
        "Solo duración": n_frames.reshape(-1, 1),
        "Segmentos + duración": np.column_stack([n_segments, n_frames]),
    }

    baselines = {}
    for nombre, X in variantes.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")
        baselines[nombre] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "fold_scores": scores.tolist(),
        }

    return {
        "stats_by_class": stats,
        "correlation_segments_duration": corr,
        "baselines": baselines,
        "n_segments": n_segments.tolist(),
        "n_frames": n_frames.tolist(),
        "labels": labels.tolist(),
    }


def main():
    res = run_segments_diagnosis()

    print("=== Número de segmentos por clase ===")
    for nombre, st in res["stats_by_class"].items():
        print(f"{nombre}: media={st['mean']:.2f}  std={st['std']:.2f}  "
              f"min={st['min']}  max={st['max']}")
    print(f"Correlación entre número de segmentos y duración: {res['correlation_segments_duration']:.2f}")

    print("\n=== Baselines triviales (accuracy, k-fold) ===")
    for nombre, b in res["baselines"].items():
        print(f"{nombre:28s}: {b['mean']:.4f} (+/- {b['std']:.4f})")


if __name__ == "__main__":
    main()
