"""Fixed-class evaluation and calibration with explicit probability validation."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, confusion_matrix)


def _labels(values, name):
    x = np.asarray(values)
    if x.ndim != 1 or len(x) == 0 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"{name} must be nonempty integer class indices")
    if (x < 0).any():
        raise ValueError(f"{name} contains negative classes")
    return x


def _probabilities(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(labels) or probs.shape[1] < 2:
        raise ValueError("Probability shape mismatch")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to 1")
    if (labels >= probs.shape[1]).any():
        raise ValueError("Label outside probability columns")
    return probs


def calibration_bins(probs, labels, n_bins=10):
    labels = _labels(labels, "labels")
    probs = _probabilities(probs, labels)
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(1)
    correct = probs.argmax(1) == labels
    assigned = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        selection = assigned == i
        count = int(selection.sum())
        rows.append(dict(lower=i / n_bins, upper=(i + 1) / n_bins, count=count,
                         confidence=float(confidence[selection].mean()) if count else None,
                         accuracy=float(correct[selection].mean()) if count else None))
    return rows


def compute_ece(probs, labels, n_bins=10):
    bins = calibration_bins(probs, labels, n_bins)
    return float(sum(row["count"] * abs(row["accuracy"] - row["confidence"])
                     for row in bins if row["count"]) / len(labels))


def compute_classification_metrics(y_true, y_pred, y_probs=None, class_names=None, n_bins=10):
    y_true, y_pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("Prediction/label length mismatch")
    probs = _probabilities(y_probs, y_true) if y_probs is not None else None
    count = len(class_names) if class_names is not None else (probs.shape[1] if probs is not None else 12)
    if (y_true >= count).any() or (y_pred >= count).any():
        raise ValueError("Label outside fixed class mapping")
    if probs is not None and probs.shape[1] != count:
        raise ValueError("Probability class count differs from mapping")
    labels = np.arange(count)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = class_names if class_names is not None else [str(i) for i in labels]
    metrics = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f.mean()), weighted_f1=float(np.average(f, weights=support)),
        macro_precision=float(p.mean()), macro_recall=float(r.mean()),
        balanced_accuracy=float(r[support > 0].mean()),
        metric_class_count=count, supported_class_count=int((support > 0).sum()),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        per_class={name: dict(f1=float(f[i]), precision=float(p[i]), recall=float(r[i]),
                              support=int(support[i])) for i, name in enumerate(names)})
    if probs is not None:
        metrics.update(ece=compute_ece(probs, y_true, n_bins),
                       brier_score=float(np.mean(np.sum((probs - np.eye(count)[y_true]) ** 2, axis=1))),
                       nll=float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)).mean()),
                       calibration_bins=calibration_bins(probs, y_true, n_bins))
    return metrics
