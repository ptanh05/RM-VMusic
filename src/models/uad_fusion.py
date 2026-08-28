"""
Uncertainty-Aware Dynamic Multimodal Fusion (UAD-Fusion) Architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import LyricsEncoder, CoverEncoder, AudioEncoder

class ReliabilityEstimator(nn.Module):
    """
    Estimates the dynamic reliability weight w_m in [0, 1] for a single modality.
    Uses modality embedding norm, variance, and binary presence mask.
    """
    def __init__(self, proj_dim=256, hidden_dim=64):
        super().__init__()
        # Input: embedding + presence mask (dim + 1)
        self.net = nn.Sequential(
            nn.Linear(proj_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, mask):
        # mask shape: [B, 1]
        x = torch.cat([emb, mask.view(-1, 1)], dim=-1)
        # Scale reliability directly by presence mask (if mask=0, reliability=0)
        raw_rel = self.net(x)
        return raw_rel * mask.view(-1, 1)

class UADFusionModel(nn.Module):
    """
    Proposed Method: Uncertainty-Aware Dynamic Multimodal Fusion under Distribution Shift.
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
        modality_dropout_p=0.20
    ):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.use_reliability = use_reliability
        self.use_modality_dropout = use_modality_dropout
        self.modality_dropout_p = modality_dropout_p

        # 1. Modality Encoders
        self.lyrics_enc = LyricsEncoder(lyrics_dim, proj_dim, dropout)
        self.cover_enc = CoverEncoder(cover_dim, proj_dim, dropout)
        self.audio_enc = AudioEncoder(audio_dim, proj_dim, dropout)

        # 2. Reliability Estimators
        self.lyrics_rel = ReliabilityEstimator(proj_dim)
        self.cover_rel = ReliabilityEstimator(proj_dim)
        self.audio_rel = ReliabilityEstimator(proj_dim)

        # 3. Dynamic Fusion Gating & Classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def apply_modality_dropout(self, mask):
        if self.training and self.use_modality_dropout:
            # Independent Bernoulli dropout on available modalities
            drop = (torch.rand_like(mask) > self.modality_dropout_p).float()
            # Ensure at least one modality remains active if any was available
            active = mask * drop
            # Fallback: if all dropped, restore original mask
            all_zero = (active.sum(dim=-1, keepdim=True) == 0) & (mask.sum(dim=-1, keepdim=True) > 0)
            active = torch.where(all_zero, mask, active)
            return active
        return mask

    def forward(self, lyrics_feat, cover_feat, audio_feat, has_lyrics, has_cover, has_audio):
        batch_size = lyrics_feat.size(0)

        # Encode modalities
        h_l = self.lyrics_enc(lyrics_feat)
        h_c = self.cover_enc(cover_feat)
        h_a = self.audio_enc(audio_feat)

        # Modality presence masks [B, 1]
        m_l = has_lyrics.view(batch_size, 1)
        m_c = has_cover.view(batch_size, 1)
        m_a = has_audio.view(batch_size, 1)

        # Apply controlled modality dropout during training
        stacked_masks = torch.cat([m_l, m_c, m_a], dim=1)
        active_masks = self.apply_modality_dropout(stacked_masks)
        m_l, m_c, m_a = active_masks[:, 0:1], active_masks[:, 1:2], active_masks[:, 2:3]

        if self.use_reliability:
            # Estimate dynamic reliability weights w_m
            w_l = self.lyrics_rel(h_l, m_l)
            w_c = self.cover_rel(h_c, m_c)
            w_a = self.audio_rel(h_a, m_a)

            # Normalize weights
            sum_w = w_l + w_c + w_a + 1e-8
            w_l_norm = w_l / sum_w
            w_c_norm = w_c / sum_w
            w_a_norm = w_a / sum_w

            # Dynamic weighted fusion
            h_fused = w_l_norm * h_l + w_c_norm * h_c + w_a_norm * h_a
            weights = torch.cat([w_l_norm, w_c_norm, w_a_norm], dim=1)
        else:
            # Simple average over available active modalities
            sum_m = m_l + m_c + m_a + 1e-8
            h_fused = (m_l * h_l + m_c * h_c + m_a * h_a) / sum_m
            weights = torch.cat([m_l / sum_m, m_c / sum_m, m_a / sum_m], dim=1)

        # Classifier logits
        logits = self.fusion_head(h_fused)

        return {
            "logits": logits,
            "fused_embedding": h_fused,
            "modality_weights": weights,
            "embeddings": {"lyrics": h_l, "cover": h_c, "audio": h_a}
        }
