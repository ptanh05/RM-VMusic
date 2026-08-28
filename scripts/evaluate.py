"""
evaluate.py
RM-VMusic Phase 4: Comprehensive Model Evaluation and Confusion Matrix Generation.
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

from train_baseline import (
    GENRES,
    GENRE2ID,
    ID2GENRE,
    MultimodalMusicDataset,
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
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(model, vectorizer, test_df, modality_mode="audio_lyrics_cover", device="cpu", split_name="test"):
    model.eval()
    audio_ext = AudioFeatureExtractor(dim=128)
    cover_ext = CoverFeatureExtractor(dim=512)
    
    test_dataset = MultimodalMusicDataset(test_df, vectorizer, audio_ext, cover_ext)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_song_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            a_mask = batch["audio_mask"].to(device)
            lyrics = batch["lyrics"].to(device)
            l_mask = batch["lyrics_mask"].to(device)
            cover = batch["cover"].to(device)
            c_mask = batch["cover_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(audio, a_mask, lyrics, l_mask, cover, c_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_song_ids.extend(batch["song_id"])
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
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
        "modality_mode": modality_mode,
        "sample_count": len(test_df),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }
    
    return results, cm

def plot_confusion_matrix(cm, filename="confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(10, 8), dpi=200)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
    print(f"[OK] Saved Confusion Matrix to {out_path}")
