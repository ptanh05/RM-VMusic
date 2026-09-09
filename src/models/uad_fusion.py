"""UAD v1: positive uncertainty, exponential reliability, masked weighted sum."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import LyricsEncoder, CoverEncoder, AudioEncoder

MODALITIES = ("lyrics", "cover", "audio")


class ReliabilityEstimator(nn.Module):
    """u=softplus(g(z)); reliability=exp(-u). This is a learned score, not a variance."""
    def __init__(self, proj_dim=256, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def uncertainty(self, emb):
        return F.softplus(self.net(emb))

    def forward(self, emb, mask):
        return torch.exp(-self.uncertainty(emb)) * mask.view(-1, 1)


class UADFusionModel(nn.Module):
    """Stable, inspectable fusion. Absent modalities have exactly zero weights.

    Rows with no available modality yield zero fused embedding; classifier bias
    produces finite prior-like logits. The model does not claim informed evidence
    on those rows. Return has_evidence so downstream evaluation can separate them.
    """
    def __init__(self, lyrics_dim=5000, cover_dim=512, audio_dim=128, proj_dim=256,
                 num_classes=12, dropout=0.30, use_reliability=True,
                 use_modality_dropout=True, modality_dropout_p=0.20):
        super().__init__()
        if not 0 <= modality_dropout_p <= 1:
            raise ValueError("modality_dropout_p must be in [0, 1]")
        self.num_classes, self.proj_dim = num_classes, proj_dim
        self.use_reliability = use_reliability
        self.use_modality_dropout = use_modality_dropout
        self.modality_dropout_p = modality_dropout_p
        # Old sigmoid/concatenation checkpoints must not silently change meaning.
        self.register_buffer("architecture_version", torch.tensor(1, dtype=torch.int64))
        self.lyrics_enc = LyricsEncoder(lyrics_dim, proj_dim, dropout)
        self.cover_enc = CoverEncoder(cover_dim, proj_dim, dropout)
        self.audio_enc = AudioEncoder(audio_dim, proj_dim, dropout)
        self.lyrics_rel = ReliabilityEstimator(proj_dim)
        self.cover_rel = ReliabilityEstimator(proj_dim)
        self.audio_rel = ReliabilityEstimator(proj_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(proj_dim, num_classes))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        version = state_dict.get("architecture_version")
        if version is None or int(version) != 1:
            raise ValueError("Legacy/incompatible UAD checkpoint; explicit migration or retraining is required")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def apply_modality_dropout(self, mask):
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Modality masks must be finite binary values")
        if self.training and self.use_modality_dropout:
            active = mask * (torch.rand_like(mask) >= self.modality_dropout_p).to(mask.dtype)
            all_dropped = (active.sum(1, keepdim=True) == 0) & (mask.sum(1, keepdim=True) > 0)
            # Preserve the established policy: restore original modalities if all drop.
            return torch.where(all_dropped, mask, active)
        return mask

    def forward(self, lyrics_feat, cover_feat, audio_feat, has_lyrics, has_cover, has_audio):
        features = (lyrics_feat, cover_feat, audio_feat)
        batch_size = lyrics_feat.shape[0]
        masks = torch.stack([x.reshape(batch_size) for x in (has_lyrics, has_cover, has_audio)], dim=1)
        if not torch.isfinite(masks).all() or not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Modality masks must be finite binary values")
        for x in features:
            if x.ndim != 2 or x.shape[0] != batch_size or not torch.isfinite(x).all():
                raise ValueError("Feature inputs must be finite [batch, dimension] tensors")
        active = self.apply_modality_dropout(masks)
        embeddings = [getattr(self, f"{mod}_enc")(x * active[:, i:i+1])
                      for i, (mod, x) in enumerate(zip(MODALITIES, features))]
        uncertainty = torch.cat([getattr(self, f"{mod}_rel").uncertainty(z)
                                 for mod, z in zip(MODALITIES, embeddings)], dim=1)
        reliability = torch.exp(-uncertainty) * active
        has_evidence = active.sum(1) > 0
        if self.use_reliability:
            scores = (-uncertainty).masked_fill(active == 0, -torch.inf)
            # Avoid softmax(-inf, -inf, -inf), and retain gradients for active rows.
            scores = torch.where(has_evidence[:, None], scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, dim=1) * active
        else:
            weights = active / active.sum(1, keepdim=True).clamp_min(1)
        fused = sum(weights[:, i:i+1] * z for i, z in enumerate(embeddings))
        return {
            "logits": self.fusion_head(fused), "fused_embedding": fused,
            "modality_weights": weights, "active_masks": active,
            "has_evidence": has_evidence,
            "uncertainty": {m: uncertainty[:, i] for i, m in enumerate(MODALITIES)},
            "reliability": {m: reliability[:, i] for i, m in enumerate(MODALITIES)},
            "fusion_weights": {m: weights[:, i] for i, m in enumerate(MODALITIES)},
            "embeddings": dict(zip(MODALITIES, embeddings)),
        }
