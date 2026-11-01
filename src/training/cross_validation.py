"""
Validación cruzada k-fold -- necesaria dado el tamaño reducido del
dataset (200 muestras), para no desperdiciar datos en un split fijo.
"""

import copy
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import TRAIN, TrainConfig
from src.datasets.yuhmu_dataset import WaveformPronunciationDataset, collate_fn
from src.evaluation.metrics import compute_metrics
from src.preprocessing.augmentation import SpecAugment


class _AugmentedSubset(Dataset):
    """Subconjunto de entrenamiento que aplica SpecAugment únicamente a los
    datos de entrenamiento, dejando intacto el subconjunto de validación."""

    def __init__(self, dataset, indices, augment):
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.augment is not None and hasattr(self.dataset, "get_item") and not isinstance(
            self.dataset, WaveformPronunciationDataset
        ):
            return self.dataset.get_item(real_idx, transform=self.augment)
        return self.dataset[real_idx]


def run_kfold(
    dataset,
    model_fn,
    device=None,
    train_cfg: TrainConfig = TRAIN,
    progress_callback=None,
    return_best_model: bool = False,
):
    """
    dataset: instancia de YuhmuPronunciationDataset, SegmentedPronunciationDataset
             o WaveformPronunciationDataset
    model_fn: función sin argumentos que devuelve un modelo nuevo
              (para reiniciar pesos en cada fold)
    train_cfg: configuración de entrenamiento (hiperparámetros y SpecAugment)
    progress_callback: función opcional callback(fold_idx, k_folds, epoch_idx, epochs, loss)
    return_best_model: si es True, devuelve (fold_results, best_model_Instance)
    """
    device = device or (
        train_cfg.device if (train_cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    labels = [label for _, label in dataset.samples]

    skf = StratifiedKFold(n_splits=train_cfg.k_folds, shuffle=True, random_state=train_cfg.seed)

    augment = None
    if train_cfg.use_spec_augment and not isinstance(dataset, WaveformPronunciationDataset):
        augment = SpecAugment(
            freq_mask_param=train_cfg.freq_mask_param,
            time_mask_param=train_cfg.time_mask_param,
        )

    fold_results = []
    best_f1 = -1.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n--- Fold {fold + 1}/{train_cfg.k_folds} ---")
        torch.manual_seed(train_cfg.seed + fold)  # reproducibilidad: pesos iniciales y orden de lotes

        train_subset = (
            _AugmentedSubset(dataset, train_idx, augment)
            if augment is not None
            else Subset(dataset, train_idx)
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = model_fn().to(device)
        n_pos = sum(labels[i] for i in train_idx)
        n_neg = len(train_idx) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=train_cfg.lr)

        loss_history = []
        for epoch in range(train_cfg.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for mels, mask, y in train_loader:
                mels, mask, y = mels.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(mels, mask)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(round(avg_loss, 5))
            if progress_callback is not None:
                progress_callback(fold + 1, train_cfg.k_folds, epoch + 1, train_cfg.epochs, avg_loss)

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
        metrics["loss_history"] = loss_history
        print({k: v for k, v in metrics.items() if k != "loss_history"})
        fold_results.append(metrics)

        if return_best_model and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = copy.deepcopy(model).cpu()

    if return_best_model:
        return fold_results, best_model
    return fold_results
