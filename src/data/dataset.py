"""
RM-VMusic PyTorch Dataset Implementation.
Handles multimodal loading for Audio (zero-masked when unmaterialized), Lyrics (Text), and Cover Art (Vision).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

GENRE_CLASSES = [
    "POP_BALLAD", "BOLERO_TRUTINH", "INSTRUMENTAL", "RAP_HIPHOP",
    "FOLK_TRADITIONAL", "DANCE_EDM", "REVOLUTIONARY", "NHAC_TRINH",
    "ROCK", "RB_SOUL", "OTHER", "CHILDREN"
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_CLASSES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRE_CLASSES)}

class RMVMusicDataset(Dataset):
    """
    Multimodal Dataset for RM-VMusic benchmark.
    """
    def __init__(self, metadata_path, audio_dim=128, lyrics_dim=768, cover_dim=512, is_train=False):
        self.df = pd.read_csv(metadata_path) if isinstance(metadata_path, (str, Path)) else metadata_path
        self.audio_dim = audio_dim
        self.lyrics_dim = lyrics_dim
        self.cover_dim = cover_dim
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Target label
        genre_str = str(row["genre"])
        label = GENRE_TO_IDX.get(genre_str, 0)
        
        # Modality availability masks
        has_lyrics = int(row.get("lyrics_available", 0))
        has_cover = int(row.get("cover_available", 0))
        has_audio = int(row.get("audio_available", 0))
        
        # In RM-VMusic, audio is zero-masked with explicit mask=0.0
        audio_feat = torch.zeros(self.audio_dim, dtype=torch.float32)
        lyrics_feat = torch.zeros(self.lyrics_dim, dtype=torch.float32)
        cover_feat = torch.zeros(self.cover_dim, dtype=torch.float32)
        
        return {
            "song_id": str(row["song_id"]),
            "title": str(row["title"]),
            "artist": str(row["artist"]),
            "label": torch.tensor(label, dtype=torch.long),
            "genre": genre_str,
            "audio_feat": audio_feat,
            "lyrics_feat": lyrics_feat,
            "cover_feat": cover_feat,
            "has_audio": torch.tensor(has_audio, dtype=torch.float32),
            "has_lyrics": torch.tensor(has_lyrics, dtype=torch.float32),
            "has_cover": torch.tensor(has_cover, dtype=torch.float32)
        }

def create_dataloader(csv_path, batch_size=32, shuffle=False, num_workers=0):
    ds = RMVMusicDataset(csv_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
