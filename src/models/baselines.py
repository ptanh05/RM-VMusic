"""
Physical Modality Baselines for RM-VMusic.
Includes: Early Concat, Late Fusion Ensemble, Lyrics-Only, Cover-Only, Audio-Only.
"""
import torch
import torch.nn as nn
from .encoders import LyricsEncoder, CoverEncoder, AudioEncoder

class EarlyConcatModel(nn.Module):
    """
    Standard Early Concatenation Fusion Baseline.
    Concatenates projected features [h_l, h_c, h_a] and passes through MLP.
    """
    def __init__(self, lyrics_dim=5000, cover_dim=512, audio_dim=128, proj_dim=256, num_classes=12, dropout=0.30):
        super().__init__()
        self.lyrics_enc = LyricsEncoder(lyrics_dim, proj_dim, dropout)
        self.cover_enc = CoverEncoder(cover_dim, proj_dim, dropout)
        self.audio_enc = AudioEncoder(audio_dim, proj_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, lyrics_feat, cover_feat, audio_feat, has_lyrics, has_cover, has_audio):
        batch_size = lyrics_feat.size(0)
        h_l = self.lyrics_enc(lyrics_feat) * has_lyrics.view(batch_size, 1)
        h_c = self.cover_enc(cover_feat) * has_cover.view(batch_size, 1)
        h_a = self.audio_enc(audio_feat) * has_audio.view(batch_size, 1)

        h_cat = torch.cat([h_l, h_c, h_a], dim=-1)
        logits = self.classifier(h_cat)
        return {"logits": logits, "fused_embedding": h_cat}

class LateFusionModel(nn.Module):
    """
    Late Fusion Ensemble Baseline: Independent classifiers per modality with logit averaging.
    """
    def __init__(self, lyrics_dim=5000, cover_dim=512, audio_dim=128, proj_dim=256, num_classes=12, dropout=0.30):
        super().__init__()
        self.lyrics_head = nn.Sequential(
            LyricsEncoder(lyrics_dim, proj_dim, dropout),
            nn.Linear(proj_dim, num_classes)
        )
        self.cover_head = nn.Sequential(
            CoverEncoder(cover_dim, proj_dim, dropout),
            nn.Linear(proj_dim, num_classes)
        )
        self.audio_head = nn.Sequential(
            AudioEncoder(audio_dim, proj_dim, dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, lyrics_feat, cover_feat, audio_feat, has_lyrics, has_cover, has_audio):
        batch_size = lyrics_feat.size(0)
        m_l = has_lyrics.view(batch_size, 1)
        m_c = has_cover.view(batch_size, 1)
        m_a = has_audio.view(batch_size, 1)

        logits_l = self.lyrics_head(lyrics_feat) * m_l
        logits_c = self.cover_head(cover_feat) * m_c
        logits_a = self.audio_head(audio_feat) * m_a

        denom = m_l + m_c + m_a + 1e-8
        avg_logits = (logits_l + logits_c + logits_a) / denom
        return {"logits": avg_logits}

class SingleModalityModel(nn.Module):
    """
    Single Modality Classifier (Lyrics-Only, Cover-Only, or Audio-Only).
    """
    def __init__(self, modality="lyrics", in_dim=5000, proj_dim=256, num_classes=12, dropout=0.30):
        super().__init__()
        self.modality = modality
        if modality == "lyrics":
            self.encoder = LyricsEncoder(in_dim, proj_dim, dropout)
        elif modality == "cover":
            self.encoder = CoverEncoder(in_dim, proj_dim, dropout)
        elif modality == "audio":
            self.encoder = AudioEncoder(in_dim, proj_dim, dropout)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, feat, mask=None):
        h = self.encoder(feat)
        if mask is not None:
            h = h * mask.view(-1, 1)
        logits = self.classifier(h)
        return {"logits": logits, "embedding": h}
