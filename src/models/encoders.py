"""
Modality Encoders for Lyrics, Cover Vision, and Audio.
"""
import torch
import torch.nn as nn

class ModalityProjector(nn.Module):
    """
    Projects raw modality features into a unified embedding dimension with LayerNorm and Dropout.
    """
    def __init__(self, in_dim, proj_dim=256, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class LyricsEncoder(nn.Module):
    def __init__(self, in_dim=5000, proj_dim=256, dropout=0.30):
        super().__init__()
        self.projector = ModalityProjector(in_dim, proj_dim, dropout)

    def forward(self, x):
        return self.projector(x)

class CoverEncoder(nn.Module):
    def __init__(self, in_dim=512, proj_dim=256, dropout=0.30):
        super().__init__()
        self.projector = ModalityProjector(in_dim, proj_dim, dropout)

    def forward(self, x):
        return self.projector(x)

class AudioEncoder(nn.Module):
    def __init__(self, in_dim=128, proj_dim=256, dropout=0.30):
        super().__init__()
        self.projector = ModalityProjector(in_dim, proj_dim, dropout)

    def forward(self, x):
        return self.projector(x)
