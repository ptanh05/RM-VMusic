"""
train_proposed.py
RM-VMusic Master Phase: Uncertainty-Aware Dynamic Multimodal Fusion under Distribution Shift (UAD-Fusion).

Architecture Components:
1. Modality-Specific Encoders (Lyrics 5000 -> 256, Cover 512 -> 256, Audio 128 -> 256)
2. Uncertainty / Reliability Estimator:
   u_m = Softplus(Linear(h_m))
   w_m = exp(-u_m) / sum_k(exp(-u_k))
3. Modality Dropout (p=0.20 during training)
4. Supervised Contrastive Representation Learning Loss (SupCon)
5. Distribution-Invariance Regularizer
6. Balanced Cross-Entropy with Training-Set Class Weights
"""

import sys
import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

GENRES_12 = [
    "POP_BALLAD",
    "BOLERO_TRUTINH",
    "INSTRUMENTAL",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "DANCE_EDM",
    "REVOLUTIONARY",
    "NHAC_TRINH",
    "ROCK",
    "RB_SOUL",
    "OTHER",
    "CHILDREN"
]
GENRE2ID = {g: i for i, g in enumerate(GENRES_12)}
ID2GENRE = {i: g for i, g in enumerate(GENRES_12)}

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss over fused multimodal representations."""
    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
            
        features = F.normalize(features, dim=1)
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        return -mean_log_prob_pos.mean()

class UADFusionModel(nn.Module):
    """
    Uncertainty-Aware Dynamic Multimodal Fusion Model (UAD-Fusion) for 12-class Genre Classification.
    """
    def __init__(
        self,
        lyrics_dim=5000,
        cover_dim=512,
        audio_dim=128,
        proj_dim=256,
        num_classes=12,
        dropout=0.30,
        use_reliability=True,
        use_modality_dropout=True,
        p_drop=0.20
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.use_reliability = use_reliability
        self.use_modality_dropout = use_modality_dropout
        self.p_drop = p_drop
        
        # 1. Modality Encoders
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        self.cover_encoder = nn.Sequential(
            nn.Linear(cover_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # 2. Modality Uncertainty Estimators (Outputs log-variance / uncertainty score)
        if self.use_reliability:
            self.lyrics_unc = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
            self.cover_unc = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
            self.audio_unc = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
            
        # 3. Dynamic Multimodal Fusion Layer
        fusion_dim = proj_dim * 3 if not use_reliability else proj_dim * 2
        self.fusion_net = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, batch, apply_modality_dropout=True):
        # Raw features & active masks
        l_feat, l_mask = batch["lyrics"], batch["lyrics_mask"].unsqueeze(1)
        c_feat, c_mask = batch["cover"], batch["cover_mask"].unsqueeze(1)
        a_feat, a_mask = batch["audio"], batch["audio_mask"].unsqueeze(1)
        
        # Modality dropout during training
        if self.training and self.use_modality_dropout and apply_modality_dropout:
            if random.random() < self.p_drop:
                l_mask = l_mask * 0.0
            if random.random() < self.p_drop:
                c_mask = c_mask * 0.0
            if random.random() < self.p_drop:
                a_mask = a_mask * 0.0
                
        # Encoders
        h_l = self.lyrics_encoder(l_feat) * l_mask
        h_c = self.cover_encoder(c_feat) * c_mask
        h_a = self.audio_encoder(a_feat) * a_mask
        
        # Reliability & Uncertainty Weighting
        if self.use_reliability:
            # Estimate raw uncertainty (softplus ensures positive variance)
            u_l = F.softplus(self.lyrics_unc(h_l)) + (1.0 - l_mask) * 10.0
            u_c = F.softplus(self.cover_unc(h_c)) + (1.0 - c_mask) * 10.0
            u_a = F.softplus(self.audio_unc(h_a)) + (1.0 - a_mask) * 10.0
            
            # Inverse variance weighting (normalized across available modalities)
            log_weights = torch.cat([-u_l, -u_c, -u_a], dim=1)
            weights = F.softmax(log_weights, dim=1) # [B, 3]
            
            w_l = weights[:, 0:1]
            w_c = weights[:, 1:2]
            w_a = weights[:, 2:3]
            
            # Weighted representations
            h_l_w = h_l * w_l
            h_c_w = h_c * w_c
            h_a_w = h_a * w_a
            
            concat_repr = torch.cat([h_l_w, h_c_w, h_a_w], dim=1)
            uncertainties = {"lyrics": u_l, "cover": u_c, "audio": u_a, "weights": weights}
        else:
            concat_repr = torch.cat([h_l, h_c, h_a], dim=1)
            weights = torch.ones((l_feat.shape[0], 3), device=l_feat.device) / 3.0
            uncertainties = {"weights": weights}
            
        fused_emb = self.fusion_net(concat_repr)
        logits = self.classifier(fused_emb)
        
        return {
            "logits": logits,
            "fused_emb": fused_emb,
            "uncertainties": uncertainties
        }

def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(np.sum(in_bin))
            })
    return float(ece), bin_stats
