"""
Evaluation metrics for RM-VMusic:
- Macro-F1
- Weighted-F1
- Accuracy
- Expected Calibration Error (ECE)
- Brier Score
- Per-class metrics
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, brier_score_loss

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

def compute_classification_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Computes comprehensive metric suite.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec
    }

    if y_probs is not None:
        metrics["ece"] = compute_ece(y_probs, y_true)
        # Brier score (multi-class one-hot)
        n_classes = y_probs.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        metrics["brier_score"] = float(np.mean(np.sum((y_probs - y_true_onehot)**2, axis=1)))

    if class_names is not None:
        per_class = {}
        for i, c_name in enumerate(class_names):
            c_mask = (y_true == i)
            if np.sum(c_mask) > 0:
                p = precision_score(y_true == i, y_pred == i, zero_division=0)
                r = recall_score(y_true == i, y_pred == i, zero_division=0)
                f = f1_score(y_true == i, y_pred == i, zero_division=0)
                per_class[c_name] = {"f1": float(f), "precision": float(p), "recall": float(r), "support": int(np.sum(c_mask))}
        metrics["per_class"] = per_class

    return metrics
