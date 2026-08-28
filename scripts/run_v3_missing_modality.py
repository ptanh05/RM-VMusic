"""
run_v3_missing_modality.py
Missing Modality Robustness Evaluation on Dataset V3 (N = 5,569).
Evaluates performance under 7 modality availability subsets: [L, C, A, LC, LA, CA, LCA].
"""
import sys
import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models.uad_fusion import UADFusionModel
from src.training.trainer import FeatureTensorDataset, compute_class_weights, train_single_model
from src.evaluation.metrics import compute_classification_metrics

SEEDS = [42, 123, 2024, 3407, 7777]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPER_DIR = REPORTS_DIR / "paper"

MODALITY_COMBINATIONS = [
    {"name": "Lyrics_Only (L)", "mask_l": 1.0, "mask_c": 0.0, "mask_a": 0.0},
    {"name": "Cover_Only (C)", "mask_l": 0.0, "mask_c": 1.0, "mask_a": 0.0},
    {"name": "Audio_Only (A)", "mask_l": 0.0, "mask_c": 0.0, "mask_a": 1.0},
    {"name": "Lyrics_Cover (LC)", "mask_l": 1.0, "mask_c": 1.0, "mask_a": 0.0},
    {"name": "Lyrics_Audio (LA)", "mask_l": 1.0, "mask_c": 0.0, "mask_a": 1.0},
    {"name": "Cover_Audio (CA)", "mask_l": 0.0, "mask_c": 1.0, "mask_a": 1.0},
    {"name": "All_Three (LCA)", "mask_l": 1.0, "mask_c": 1.0, "mask_a": 1.0}
]

def load_features():
    with open(FEATURES_DIR / "song_id_index_map.pkl", "rb") as f:
        song_id_map = pickle.load(f)
    lyrics_feats = np.load(FEATURES_DIR / "lyrics" / "lyrics_features_5000.npy")
    lyrics_masks = np.load(FEATURES_DIR / "lyrics" / "lyrics_masks.npy")
    cover_feats = np.load(FEATURES_DIR / "cover" / "cover_features_512.npy")
    cover_masks = np.load(FEATURES_DIR / "cover" / "cover_masks.npy")
    audio_feats = np.load(FEATURES_DIR / "audio" / "audio_features_128.npy")
    audio_masks = np.load(FEATURES_DIR / "audio" / "audio_masks.npy")
    return lyrics_feats, cover_feats, audio_feats, lyrics_masks, cover_masks, audio_masks, song_id_map

def run_missing_modality_eval():
    print("=== RM-VMusic: Missing Modality Robustness Evaluation on Dataset V3 ===")
    lyrics_feats, cover_feats, audio_feats, lyrics_masks, cover_masks, audio_masks, song_id_map = load_features()

    tr_df = pd.read_csv(SPLITS_DIR / "iid" / "train.csv")
    va_df = pd.read_csv(SPLITS_DIR / "iid" / "val.csv")
    te_df = pd.read_csv(SPLITS_DIR / "missing_modality" / "test.csv")

    train_ds = FeatureTensorDataset(tr_df, lyrics_feats, cover_feats, audio_feats, lyrics_masks, cover_masks, audio_masks, song_id_map)
    val_ds = FeatureTensorDataset(va_df, lyrics_feats, cover_feats, audio_feats, lyrics_masks, cover_masks, audio_masks, song_id_map)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    class_weights = compute_class_weights(train_ds.labels, num_classes=12)

    # Train model on seed 42
    model = UADFusionModel(lyrics_dim=5000, cover_dim=512, audio_dim=128, proj_dim=256, num_classes=12, use_reliability=True, use_modality_dropout=True)
    train_single_model(model, train_loader, val_loader, val_loader, epochs=25, lr=0.001, patience=6, class_weights=class_weights, is_proposed=True)

    results = []
    indices = [song_id_map[sid] for sid in te_df["song_id"] if sid in song_id_map]
    indices_arr = np.array(indices, dtype=np.int64)

    test_l_f = torch.tensor(lyrics_feats[indices_arr], dtype=torch.float32)
    test_c_f = torch.tensor(cover_feats[indices_arr], dtype=torch.float32)
    test_a_f = torch.tensor(audio_feats[indices_arr], dtype=torch.float32)
    test_labels = np.array([tr_df["genre"].unique().tolist().index(g) if g in tr_df["genre"].unique().tolist() else 0 for g in te_df["genre"]])

    model.eval()
    with torch.no_grad():
        for combo in MODALITY_COMBINATIONS:
            m_name = combo["name"]
            m_l = torch.full((len(te_df),), combo["mask_l"], dtype=torch.float32)
            m_c = torch.full((len(te_df),), combo["mask_c"], dtype=torch.float32)
            m_a = torch.full((len(te_df),), combo["mask_a"], dtype=torch.float32)

            out = model(test_l_f, test_c_f, test_a_f, m_l, m_c, m_a)
            probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            metrics = compute_classification_metrics(test_labels, preds, probs)
            print(f"Combination {m_name:20s} | Macro-F1: {metrics['macro_f1']*100:.2f}% | Acc: {metrics['accuracy']*100:.2f}% | ECE: {metrics['ece']:.4f}")

            results.append({
                "Modality_Subset": m_name,
                "Macro_F1": round(metrics["macro_f1"] * 100, 2),
                "Accuracy": round(metrics["accuracy"] * 100, 2),
                "ECE": round(metrics["ece"], 4)
            })

    df_mm = pd.DataFrame(results)
    df_mm.to_csv(PAPER_DIR / "paper_missing_modality_v3.csv", index=False)
    print("Saved paper_missing_modality_v3.csv successfully.")

if __name__ == "__main__":
    run_missing_modality_eval()
