"""Shared training/evaluation contracts. Imported datasets retain their public name."""
import copy
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import GENRE_CLASSES, GENRE_TO_IDX, MODALITIES
from ..data.dataset import FeatureTensorDataset
from .losses import WeightedCrossEntropyLoss, SupervisedContrastiveLoss
from ..evaluation.metrics import compute_classification_metrics


def set_seed(seed, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_class_weights(labels_tensor, num_classes=12):
    if labels_tensor.ndim != 1 or len(labels_tensor) == 0 or labels_tensor.dtype != torch.long:
        raise ValueError("Class weights require nonempty int64 train labels")
    if (labels_tensor < 0).any() or (labels_tensor >= num_classes).any():
        raise ValueError("Train labels outside class mapping")
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    weights = len(labels_tensor) / (num_classes * (counts + 1.0))
    return weights / weights.sum() * num_classes


def batch_inputs(batch, device="cpu"):
    return [batch[f"{m}_feat"].to(device) for m in MODALITIES] + [
        batch[f"has_{m}"].to(device) for m in MODALITIES]


def predict_model(model, loader, device="cpu", missing_rate=0.0, mask_seed=42):
    """Mask draws depend on seed and row order, never on model RNG or batch size."""
    if not 0 <= missing_rate <= 1:
        raise ValueError("Invalid missing rate")
    model.to(device)
    model.eval()
    labels, probs, ids, masks, weights = [], [], [], [], []
    rng = np.random.default_rng(mask_seed)
    with torch.no_grad():
        for batch in loader:
            inputs = batch_inputs(batch, device)
            if missing_rate:
                keep = torch.tensor(rng.random((len(inputs[0]), 3)) >= missing_rate,
                                    dtype=inputs[0].dtype, device=device)
                for i in range(3):
                    inputs[i + 3] = inputs[i + 3] * keep[:, i]
            out = model(*inputs)
            probabilities = torch.softmax(out["logits"], dim=-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite model probabilities")
            labels.extend(batch["label"].numpy())
            probs.extend(probabilities.cpu().numpy())
            ids.extend(batch.get("song_id", []))
            masks.extend(torch.stack(inputs[3:], dim=1).cpu().numpy())
            if "modality_weights" in out:
                weights.extend(out["modality_weights"].cpu().numpy())
    if not labels:
        raise ValueError("Evaluation loader is empty")
    return dict(labels=np.asarray(labels, dtype=np.int64), probabilities=np.asarray(probs),
                song_ids=ids, masks=np.asarray(masks), weights=np.asarray(weights))


def evaluate_model(model, loader, device="cpu", ece_bins=10):
    result = predict_model(model, loader, device)
    return compute_classification_metrics(
        result["labels"], result["probabilities"].argmax(1), result["probabilities"],
        class_names=GENRE_CLASSES, n_bins=ece_bins)


def fit_model(model, train_loader, val_loader, *, epochs=35, lr=0.001,
              weight_decay=1e-4, patience=8, class_weights=None, device="cpu",
              is_proposed=True, lambda_supcon=0.15, lambda_inv=0.0, ece_bins=10):
    """Select exclusively on validation Macro-F1. No test loader is accepted."""
    if lambda_inv != 0:
        raise ValueError("Unpaired invariance loss is retired; lambda_inv must be zero")
    if epochs < 1 or patience < 1 or len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training requires positive epochs/patience and nonempty loaders")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = WeightedCrossEntropyLoss(class_weights.to(device) if class_weights is not None else None)
    supcon = SupervisedContrastiveLoss(temperature=0.10)
    best_f1, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            inputs = batch_inputs(batch, device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(*inputs)
            loss = criterion(out["logits"], labels)
            if is_proposed and lambda_supcon:
                available = out["has_evidence"]
                loss = loss + lambda_supcon * supcon(out["fused_embedding"][available], labels[available])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        val = evaluate_model(model, val_loader, device, ece_bins)
        score = val["macro_f1"]
        scheduler.step(score)
        history.append(dict(epoch=epoch, train_loss=total / len(train_loader), val_macro_f1=score))
        if score > best_f1:
            best_f1, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_macro_f1": best_f1, "history": history}


def train_single_model(model, train_loader, val_loader, test_loader, epochs=35, lr=0.001,
                       weight_decay=1e-4, patience=8, class_weights=None, device="cpu",
                       is_proposed=True, lambda_supcon=0.15, lambda_inv=0.0):
    """Compatibility wrapper: fit on train/val, then evaluate once on test."""
    fit_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
              weight_decay=weight_decay, patience=patience, class_weights=class_weights,
              device=device, is_proposed=is_proposed, lambda_supcon=lambda_supcon, lambda_inv=lambda_inv)
    return evaluate_model(model, test_loader, device)
