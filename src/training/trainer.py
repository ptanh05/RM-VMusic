"""
RM-VMusic Master Trainer.
Handles training, validation, multi-seed orchestration, and evaluation across distribution shift scenarios.
"""
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from ..models.uad_fusion import UADFusionModel
from ..models.baselines import EarlyConcatModel, LateFusionModel, SingleModalityModel
from ..training.losses import WeightedCrossEntropyLoss, SupervisedContrastiveLoss, DistributionInvarianceLoss
from ..evaluation.metrics import compute_classification_metrics

GENRE_CLASSES = [
    "POP_BALLAD", "BOLERO_TRUTINH", "INSTRUMENTAL", "RAP_HIPHOP",
    "FOLK_TRADITIONAL", "DANCE_EDM", "REVOLUTIONARY", "NHAC_TRINH",
    "ROCK", "RB_SOUL", "OTHER", "CHILDREN"
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_CLASSES)}

class FeatureTensorDataset(Dataset):
    """
    In-memory tensor dataset indexing pre-extracted features.
    """
    def __init__(self, df, lyrics_feats, cover_feats, audio_feats, lyrics_masks, cover_masks, audio_masks, song_id_map):
        self.df = df
        indices = [song_id_map[sid] for sid in df["song_id"] if sid in song_id_map]
        self.indices = np.array(indices, dtype=np.int64)

        self.lyrics_feats = torch.tensor(lyrics_feats[self.indices], dtype=torch.float32)
        self.cover_feats = torch.tensor(cover_feats[self.indices], dtype=torch.float32)
        self.audio_feats = torch.tensor(audio_feats[self.indices], dtype=torch.float32)

        self.lyrics_masks = torch.tensor(lyrics_masks[self.indices], dtype=torch.float32)
        self.cover_masks = torch.tensor(cover_masks[self.indices], dtype=torch.float32)
        self.audio_masks = torch.tensor(audio_masks[self.indices], dtype=torch.float32)

        labels = [GENRE_TO_IDX.get(str(g), 0) for g in df["genre"]]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "lyrics_feat": self.lyrics_feats[idx],
            "cover_feat": self.cover_feats[idx],
            "audio_feat": self.audio_feats[idx],
            "has_lyrics": self.lyrics_masks[idx],
            "has_cover": self.cover_masks[idx],
            "has_audio": self.audio_masks[idx],
            "label": self.labels[idx]
        }

def compute_class_weights(labels_tensor, num_classes=12):
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    total = len(labels_tensor)
    weights = total / (num_classes * (counts + 1.0))
    return weights / weights.sum() * num_classes

def train_single_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=35,
    lr=0.001,
    weight_decay=1e-4,
    patience=8,
    class_weights=None,
    device="cpu",
    is_proposed=True,
    lambda_supcon=0.15,
    lambda_inv=0.05
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    ce_criterion = WeightedCrossEntropyLoss(class_weights.to(device) if class_weights is not None else None)
    supcon_criterion = SupervisedContrastiveLoss(temperature=0.10)
    inv_criterion = DistributionInvarianceLoss()

    best_val_f1 = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            l_f = batch["lyrics_feat"].to(device)
            c_f = batch["cover_feat"].to(device)
            a_f = batch["audio_feat"].to(device)
            m_l = batch["has_lyrics"].to(device)
            m_c = batch["has_cover"].to(device)
            m_a = batch["has_audio"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(l_f, c_f, a_f, m_l, m_c, m_a)
            logits = out["logits"]

            loss = ce_criterion(logits, labels)

            if is_proposed and "fused_embedding" in out:
                fused_emb = out["fused_embedding"]
                sup_loss = supcon_criterion(fused_emb, labels)
                mod_masks = torch.cat([m_l.view(-1, 1), m_c.view(-1, 1), m_a.view(-1, 1)], dim=1)
                inv_loss = inv_criterion(fused_emb, mod_masks)
                loss = loss + lambda_supcon * sup_loss + lambda_inv * inv_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_metrics = evaluate_model(model, val_loader, device=device)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_metrics = evaluate_model(model, test_loader, device=device)
    return test_metrics

def evaluate_model(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            l_f = batch["lyrics_feat"].to(device)
            c_f = batch["cover_feat"].to(device)
            a_f = batch["audio_feat"].to(device)
            m_l = batch["has_lyrics"].to(device)
            m_c = batch["has_cover"].to(device)
            m_a = batch["has_audio"].to(device)
            labels = batch["label"].to(device)

            out = model(l_f, c_f, a_f, m_l, m_c, m_a)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), class_names=GENRE_CLASSES
    )
