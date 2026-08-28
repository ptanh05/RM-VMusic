"""
evaluate_proposed.py
RM-VMusic Phase 5: Evaluation Engine, Uncertainty Analysis, and Diagnostic Figure Generator.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch
from torch.utils.data import DataLoader

from train_proposed import (
    GENRES,
    GENRE2ID,
    ID2GENRE,
    ProposedMultimodalDataset,
    AudioFeatureExtractor,
    CoverFeatureExtractor
)

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "outputs" / "metrics" / "proposed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_proposed_model(model, vectorizer, test_df, device="cpu", split_name="test", mask_mode="none"):
    """
    Evaluates proposed model, extracts per-sample reliability weights and predictions.
    mask_mode allows simulating controlled missing modalities during evaluation:
    'none', 'no_audio', 'no_lyrics', 'no_cover', 'no_audio_lyrics', 'no_audio_cover', 'no_lyrics_cover'
    """
    model.eval()
    audio_ext = AudioFeatureExtractor(dim=128)
    cover_ext = CoverFeatureExtractor(dim=512)
    
    test_dataset = ProposedMultimodalDataset(test_df, vectorizer, audio_ext, cover_ext)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_weights = []
    all_uncs = []
    all_correct = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            a_mask = batch["audio_mask"].to(device)
            lyrics = batch["lyrics"].to(device)
            l_mask = batch["lyrics_mask"].to(device)
            cover = batch["cover"].to(device)
            c_mask = batch["cover_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Apply evaluation simulated mask mode
            if mask_mode == "no_audio":
                a_mask = torch.zeros_like(a_mask)
            elif mask_mode == "no_lyrics":
                l_mask = torch.zeros_like(l_mask)
            elif mask_mode == "no_cover":
                c_mask = torch.zeros_like(c_mask)
            elif mask_mode == "no_audio_lyrics":
                a_mask = torch.zeros_like(a_mask)
                l_mask = torch.zeros_like(l_mask)
            elif mask_mode == "no_audio_cover":
                a_mask = torch.zeros_like(a_mask)
                c_mask = torch.zeros_like(c_mask)
            elif mask_mode == "no_lyrics_cover":
                l_mask = torch.zeros_like(l_mask)
                c_mask = torch.zeros_like(c_mask)
                
            logits, z_fused, weights, uncs, _ = model(audio, a_mask, lyrics, l_mask, cover, c_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            lbls = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(lbls)
            all_weights.extend(weights.cpu().numpy())
            all_uncs.extend(uncs.cpu().numpy())
            all_correct.extend((preds == lbls).astype(int))
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)
    all_uncs = np.array(all_uncs)
    all_correct = np.array(all_correct)
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(GENRES)), zero_division=0)
    
    per_class_metrics = {}
    for i, g in enumerate(GENRES):
        per_class_metrics[g] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i])
        }
        
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(GENRES)))
    
    results = {
        "split_name": split_name,
        "mask_mode": mask_mode,
        "sample_count": len(test_df),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "mean_weights": {
            "audio": float(np.mean(all_weights[:, 0])),
            "lyrics": float(np.mean(all_weights[:, 1])),
            "cover": float(np.mean(all_weights[:, 2]))
        },
        "mean_uncertainties": {
            "audio": float(np.mean(all_uncs[:, 0])),
            "lyrics": float(np.mean(all_uncs[:, 1])),
            "cover": float(np.mean(all_uncs[:, 2]))
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }
    
    return results, cm, (all_weights, all_uncs, all_correct)

def plot_proposed_confusion_matrix(cm, filename="proposed_confusion_iid.png", title="Confusion Matrix"):
    plt.figure(figsize=(10, 8), dpi=200)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(title, fontsize=14, pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(GENRES))
    plt.xticks(tick_marks, GENRES, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, GENRES, fontsize=9)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, "d"),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=8,
                     color="white" if val > thresh else "black")
                     
    plt.ylabel("True Ground Truth Label", fontsize=11)
    plt.xlabel("Predicted Model Label", fontsize=11)
    plt.tight_layout()
    
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Proposed Confusion Matrix to {out_path}")

def plot_reliability_diagnostics(weights_dict, correctness_dict, ablation_f1_dict, missing_modality_dict):
    """Plots the 4 core diagnostic charts for uncertainty analysis and ablation comparison."""
    # 1. Reliability Weights Bar Chart
    plt.figure(figsize=(8, 5), dpi=200)
    modalities = ["Audio", "Lyrics", "Cover Art"]
    means = [weights_dict["audio"], weights_dict["lyrics"], weights_dict["cover"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    
    bars = plt.bar(modalities, means, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    plt.title("Learned Modality Dynamic Reliability Attention Weights (Alpha)", fontsize=12, pad=12)
    plt.ylabel("Normalized Weight (Mean Alpha)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f} ({yval*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reliability_weights.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved reliability_weights.png")

    # 2. Reliability vs Correctness
    plt.figure(figsize=(7, 5), dpi=200)
    cats = ["Correct Predictions", "Incorrect Predictions"]
    c_weights = [correctness_dict["correct_lyrics_weight"], correctness_dict["incorrect_lyrics_weight"]]
    plt.bar(cats, c_weights, color=["#2ca02c", "#d62728"], width=0.45, edgecolor="black")
    plt.title("Lyrics Modality Reliability Weight vs Prediction Correctness", fontsize=12, pad=12)
    plt.ylabel("Mean Lyrics Weight (Alpha)", fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(c_weights):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reliability_vs_correctness.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved reliability_vs_correctness.png")

    # 3. Modality Dropout Missing Modality Resilience
    plt.figure(figsize=(10, 5), dpi=200)
    modes = list(missing_modality_dict.keys())
    f1_vals = [v * 100 for v in missing_modality_dict.values()]
    plt.barh(modes, f1_vals, color="#3498db", edgecolor="black")
    plt.title("Proposed Model Robustness across Simulated Missing Modality Subsets", fontsize=12, pad=12)
    plt.xlabel("Macro-F1 Score (%)", fontsize=11)
    plt.xlim(0, 45)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    for i, v in enumerate(f1_vals):
        plt.text(v + 0.5, i, f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modality_dropout_results.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved modality_dropout_results.png")

    # 4. Ablation Macro-F1 Ladder Comparison
    plt.figure(figsize=(10, 5.5), dpi=200)
    ablations = list(ablation_f1_dict.keys())
    scores = [v * 100 for v in ablation_f1_dict.values()]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    bars = plt.bar(ablations, scores, color=colors[:len(ablations)], width=0.55, edgecolor="black")
    plt.title("Ablation Ladder Comparison (Models A -> E) on IID Benchmark", fontsize=13, pad=12)
    plt.ylabel("Macro-F1 Score (%)", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylim(0, 40)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_macro_f1.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ablation_macro_f1.png")
