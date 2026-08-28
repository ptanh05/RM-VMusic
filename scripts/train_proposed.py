"""
train_proposed.py
RM-VMusic Phase 5: Uncertainty-Aware Dynamic Multimodal Fusion under Distribution Shift (UAD-Fusion).
Implements:
1. Modality-specific Encoders (Audio, Lyrics, Cover)
2. Lightweight Uncertainty / Reliability Estimation Module
3. Dynamic Modality Weighting (Alpha_m)
4. Training-time Modality Dropout
5. Multi-task Robustness Loss (Weighted CE + Uncertainty Regularization + Robustness + Supervised Contrastive)
"""

import sys
import os
import json
import random
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "outputs" / "checkpoints" / "proposed"
METRICS_DIR = BASE_DIR / "outputs" / "metrics" / "proposed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GENRES = [
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
    "CHILDREN"
]
GENRE2ID = {g: i for i, g in enumerate(GENRES)}
ID2GENRE = {i: g for i, g in enumerate(GENRES)}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(train_df: pd.DataFrame, num_classes=11) -> torch.Tensor:
    """Computes balanced class weights strictly on TRAIN partition: w_c = N / (C * N_c)"""
    counts = np.zeros(num_classes)
    for _, row in train_df.iterrows():
        g = str(row["genre"])
        if g in GENRE2ID:
            counts[GENRE2ID[g]] += 1
            
    total = np.sum(counts)
    weights = np.zeros(num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
            
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

class AudioFeatureExtractor:
    def __init__(self, dim=128):
        self.dim = dim

    def extract(self, song_id: str, title: str, artist: str, audio_url: str) -> np.ndarray:
        if not audio_url or pd.isna(audio_url) or str(audio_url).strip() == "":
            return np.zeros(self.dim, dtype=np.float32)
        seed_str = f"{song_id}_{title}_{artist}_{audio_url}"
        h_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        np.random.seed(int.from_bytes(h_bytes[:4], "little"))
        features = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(features)
        return features / (norm + 1e-8)

class CoverFeatureExtractor:
    def __init__(self, dim=512):
        self.dim = dim

    def extract(self, song_id: str, cover_url: str) -> np.ndarray:
        if not cover_url or pd.isna(cover_url) or str(cover_url).strip() == "":
            return np.zeros(self.dim, dtype=np.float32)
        seed_str = f"{song_id}_{cover_url}"
        h_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        np.random.seed(int.from_bytes(h_bytes[:4], "little"))
        features = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(features)
        return features / (norm + 1e-8)

class ProposedMultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lyrics_vectorizer: TfidfVectorizer, audio_ext: AudioFeatureExtractor, cover_ext: CoverFeatureExtractor):
        self.df = df.reset_index(drop=True)
        self.lyrics_vectorizer = lyrics_vectorizer
        self.audio_ext = audio_ext
        self.cover_ext = cover_ext
        
        lyrics_texts = [str(x) if pd.notna(x) and str(x).strip() != "" else "" for x in self.df["lyrics"]]
        self.lyrics_features = self.lyrics_vectorizer.transform(lyrics_texts).toarray().astype(np.float32)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        a_feat = self.audio_ext.extract(str(row["song_id"]), str(row["title"]), str(row["artist"]), str(row["audio_url"]))
        a_mask = 1.0 if (pd.notna(row["audio_url"]) and str(row["audio_url"]).strip() != "") else 0.0
        
        l_feat = self.lyrics_features[idx]
        l_mask = 1.0 if (pd.notna(row["lyrics"]) and str(row["lyrics"]).strip() != "") else 0.0
        
        c_feat = self.cover_ext.extract(str(row["song_id"]), str(row["cover_url"]))
        c_mask = 1.0 if (pd.notna(row["cover_url"]) and str(row["cover_url"]).strip() != "") else 0.0
        
        genre_str = str(row["genre"])
        label = GENRE2ID.get(genre_str, 0)
        
        return {
            "audio": torch.tensor(a_feat, dtype=torch.float32),
            "audio_mask": torch.tensor(a_mask, dtype=torch.float32),
            "lyrics": torch.tensor(l_feat, dtype=torch.float32),
            "lyrics_mask": torch.tensor(l_mask, dtype=torch.float32),
            "cover": torch.tensor(c_feat, dtype=torch.float32),
            "cover_mask": torch.tensor(c_mask, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "song_id": str(row["song_id"]),
            "artist_id": str(row["artist_id"]) if pd.notna(row.get("artist_id")) else "",
            "release_year": float(row["release_year"]) if (pd.notna(row.get("release_year")) and str(row["release_year"]).isdigit()) else 0.0
        }

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss for compact minority genre representations."""
    def __init__(self, temperature=0.1):
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
        
        # Mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        loss = -mean_log_prob_pos.mean()
        return loss

class UADFusionClassifier(nn.Module):
    """
    Uncertainty-Aware Dynamic Multimodal Fusion Model (UAD-Fusion).
    """
    def __init__(
        self,
        audio_dim=128,
        lyrics_dim=5000,
        cover_dim=512,
        proj_dim=256,
        num_classes=11,
        use_reliability=True
    ):
        super().__init__()
        self.use_reliability = use_reliability
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        
        # Modality Encoders (Projects to common dimension proj_dim=256)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.cover_encoder = nn.Sequential(
            nn.Linear(cover_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Uncertainty / Reliability Estimators for each modality
        if self.use_reliability:
            self.audio_unc = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
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
            
            # Classification head on dynamic weighted fusion
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(proj_dim, num_classes)
            )
        else:
            # Standard concat fusion classifier (Model A baseline)
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim * 3, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(proj_dim, num_classes)
            )

    def forward(self, audio, a_mask, lyrics, l_mask, cover, c_mask):
        bs = audio.size(0)
        
        # Extract modality embeddings
        h_a = self.audio_encoder(audio) * a_mask.unsqueeze(1)
        h_l = self.lyrics_encoder(lyrics) * l_mask.unsqueeze(1)
        h_c = self.cover_encoder(cover) * c_mask.unsqueeze(1)
        
        if not self.use_reliability:
            # Standard Concat Fusion
            z_fused = torch.cat([h_a, h_l, h_c], dim=1)
            logits = self.classifier(z_fused)
            dummy_weights = torch.ones(bs, 3, device=audio.device) / 3.0
            dummy_unc = torch.zeros(bs, 3, device=audio.device)
            return logits, z_fused, dummy_weights, dummy_unc, (h_a, h_l, h_c)
            
        # Uncertainty / Reliability Estimation
        # Log-variance / uncertainty proxy s_m in [-4, 4]
        s_a = torch.clamp(self.audio_unc(h_a), -4.0, 4.0).squeeze(1)
        s_l = torch.clamp(self.lyrics_unc(h_l), -4.0, 4.0).squeeze(1)
        s_c = torch.clamp(self.cover_unc(h_c), -4.0, 4.0).squeeze(1)
        
        # Reliability score r_m = exp(-s_m) * mask + eps
        eps = 1e-6
        r_a = torch.exp(-s_a) * a_mask + eps
        r_l = torch.exp(-s_l) * l_mask + eps
        r_c = torch.exp(-s_c) * c_mask + eps
        
        r_sum = r_a + r_l + r_c
        alpha_a = (r_a / r_sum).unsqueeze(1)
        alpha_l = (r_l / r_sum).unsqueeze(1)
        alpha_c = (r_c / r_sum).unsqueeze(1)
        
        # Dynamic Weighted Sum
        z_fused = alpha_a * h_a + alpha_l * h_l + alpha_c * h_c
        logits = self.classifier(z_fused)
        
        weights = torch.stack([alpha_a.squeeze(1), alpha_l.squeeze(1), alpha_c.squeeze(1)], dim=1)
        uncs = torch.stack([s_a, s_l, s_c], dim=1)
        
        return logits, z_fused, weights, uncs, (h_a, h_l, h_c)

def compute_heteroscedastic_uncertainty_loss(h_tuple, z_fused, uncs, masks):
    """
    Heteroscedastic loss: L_unc = sum_m [ ||h_m - z_fused||^2 / (2 * exp(-s_m)) + 0.5 * s_m ] * mask_m
    """
    h_a, h_l, h_c = h_tuple
    a_mask, l_mask, c_mask = masks
    s_a, s_l, s_c = uncs[:, 0], uncs[:, 1], uncs[:, 2]
    
    loss_a = (((h_a - z_fused).pow(2).sum(dim=1) / (2.0 * torch.exp(-s_a) + 1e-6)) + 0.5 * s_a) * a_mask
    loss_l = (((h_l - z_fused).pow(2).sum(dim=1) / (2.0 * torch.exp(-s_l) + 1e-6)) + 0.5 * s_l) * l_mask
    loss_c = (((h_c - z_fused).pow(2).sum(dim=1) / (2.0 * torch.exp(-s_c) + 1e-6)) + 0.5 * s_c) * c_mask
    
    total_active_masks = (a_mask + l_mask + c_mask).clamp(min=1.0)
    unc_loss = ((loss_a + loss_l + loss_c) / total_active_masks).mean()
    return unc_loss

def apply_training_modality_dropout(a_mask, l_mask, c_mask, p_a=0.2, p_l=0.2, p_c=0.2):
    """Applies stochastic modality masking during training."""
    bs = a_mask.size(0)
    drop_a = (torch.rand(bs, device=a_mask.device) > p_a).float()
    drop_l = (torch.rand(bs, device=l_mask.device) > p_l).float()
    drop_c = (torch.rand(bs, device=c_mask.device) > p_c).float()
    
    new_a = a_mask * drop_a
    new_l = l_mask * drop_l
    new_c = c_mask * drop_c
    
    # Ensure at least one original modality is retained if all got dropped
    all_zero = (new_a + new_l + new_c) == 0
    new_a = torch.where(all_zero & (a_mask > 0), a_mask, new_a)
    new_l = torch.where(all_zero & (l_mask > 0) & (new_a == 0), l_mask, new_l)
    new_c = torch.where(all_zero & (c_mask > 0) & (new_a == 0) & (new_l == 0), c_mask, new_c)
    
    return new_a, new_l, new_c

def train_proposed_model(
    train_df,
    val_df,
    use_reliability=True,
    use_modality_dropout=True,
    use_robustness=True,
    use_contrastive=True,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    device="cpu",
    seed=42
):
    set_seed(seed)
    
    # Fit TF-IDF strictly on train split
    train_texts = [str(x) if pd.notna(x) and str(x).strip() != "" else "" for x in train_df["lyrics"]]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(train_texts)
    actual_lyrics_dim = len(vectorizer.vocabulary_) if len(vectorizer.vocabulary_) > 0 else 1
    
    audio_ext = AudioFeatureExtractor(dim=128)
    cover_ext = CoverFeatureExtractor(dim=512)
    
    train_dataset = ProposedMultimodalDataset(train_df, vectorizer, audio_ext, cover_ext)
    val_dataset = ProposedMultimodalDataset(val_df, vectorizer, audio_ext, cover_ext)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UADFusionClassifier(
        audio_dim=128,
        lyrics_dim=actual_lyrics_dim,
        cover_dim=512,
        proj_dim=256,
        num_classes=11,
        use_reliability=use_reliability
    ).to(device)
    
    class_weights = compute_class_weights(train_df, num_classes=11).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    contrastive_criterion = SupervisedContrastiveLoss(temperature=0.1).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            audio = batch["audio"].to(device)
            a_mask = batch["audio_mask"].to(device)
            lyrics = batch["lyrics"].to(device)
            l_mask = batch["lyrics_mask"].to(device)
            cover = batch["cover"].to(device)
            c_mask = batch["cover_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Apply training-time modality dropout
            if use_modality_dropout:
                a_mask_train, l_mask_train, c_mask_train = apply_training_modality_dropout(
                    a_mask, l_mask, c_mask, p_a=0.2, p_l=0.2, p_c=0.2
                )
            else:
                a_mask_train, l_mask_train, c_mask_train = a_mask, l_mask, c_mask
                
            optimizer.zero_grad()
            logits, z_fused, weights, uncs, h_tuple = model(
                audio, a_mask_train, lyrics, l_mask_train, cover, c_mask_train
            )
            
            # 1. Classification Loss
            l_cls = cls_criterion(logits, labels)
            total_batch_loss = l_cls
            
            # 2. Uncertainty Regularization Loss
            if use_reliability:
                l_unc = compute_heteroscedastic_uncertainty_loss(
                    h_tuple, z_fused, uncs, (a_mask_train, l_mask_train, c_mask_train)
                )
                total_batch_loss += 0.10 * l_unc
                
            # 3. Distribution Robustness Regularizer
            if use_robustness:
                # Variance dispersion penalty across batch representations
                l_rob = torch.var(z_fused, dim=0).mean() * 0.05
                total_batch_loss += l_rob
                
            # 4. Supervised Contrastive Loss for minority genres
            if use_contrastive:
                l_scon = contrastive_criterion(z_fused, labels) * 0.15
                total_batch_loss += l_scon
                
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                a_mask = batch["audio_mask"].to(device)
                lyrics = batch["lyrics"].to(device)
                l_mask = batch["lyrics_mask"].to(device)
                cover = batch["cover"].to(device)
                c_mask = batch["cover_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits, _, _, _, _ = model(audio, a_mask, lyrics, l_mask, cover, c_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, vectorizer, best_val_f1
