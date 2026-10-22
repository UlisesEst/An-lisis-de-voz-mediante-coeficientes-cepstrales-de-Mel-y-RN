"""
Validación cruzada k-fold -- necesaria dado el tamaño reducido del
dataset (200 muestras), para no desperdiciar datos en un split fijo.
"""

import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.config import TRAIN
from src.datasets.yuhmu_dataset import collate_fn
from src.evaluation.metrics import compute_metrics


def run_kfold(dataset, model_fn, device=None):
    """
    dataset: instancia de YuhmuPronunciationDataset
    model_fn: función sin argumentos que devuelve un modelo nuevo
              (para reiniciar pesos en cada fold)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels = [label for _, label in dataset.samples]

    skf = StratifiedKFold(n_splits=TRAIN.k_folds, shuffle=True, random_state=TRAIN.seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n--- Fold {fold + 1}/{TRAIN.k_folds} ---")
        torch.manual_seed(TRAIN.seed + fold)  # reproducibilidad: pesos iniciales y orden de lotes

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=TRAIN.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=TRAIN.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = model_fn().to(device)
        n_pos = sum(labels[i] for i in train_idx)
        n_neg = len(train_idx) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.lr)

        for epoch in range(TRAIN.epochs):
            model.train()
            for mels, mask, y in train_loader:
                mels, mask, y = mels.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(mels, mask)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for mels, mask, y in val_loader:
                mels, mask = mels.to(device), mask.to(device)
                logits = model(mels, mask)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu()
                y_true.extend(y.long().tolist())
                y_pred.extend(preds.tolist())

        metrics = compute_metrics(y_true, y_pred)
        print(metrics)
        fold_results.append(metrics)

    return fold_results
