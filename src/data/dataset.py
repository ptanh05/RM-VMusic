"""Aligned physical-feature datasets, with compatibility imports for old callers."""
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..config import GENRE_CLASSES, GENRE_TO_IDX, IDX_TO_GENRE, FEATURE_DIMS, MODALITIES
from .validation import validate_metadata, read_metadata


def validate_arrays(features, masks, song_id_map, dimensions=FEATURE_DIMS):
    n = len(song_id_map)
    if any(not isinstance(sid, str) or not sid for sid in song_id_map):
        raise ValueError("Feature map IDs must be nonempty strings")
    if any(type(i) not in (int, np.int64, np.int32) for i in song_id_map.values()) or set(song_id_map.values()) != set(range(n)):
        raise ValueError("Feature map must be a bijection onto contiguous row indices")
    for mod in MODALITIES:
        array, mask = np.asarray(features[mod]), np.asarray(masks[mod])
        if array.shape != (n, dimensions[mod]) or mask.shape != (n,):
            raise ValueError(f"{mod}: invalid feature/mask dimensions or row count")
        if not np.isfinite(array).all() or not np.isfinite(mask).all():
            raise ValueError(f"{mod}: non-finite (NaN/Inf) feature/mask")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{mod}: masks must be binary")
        if np.any(array[mask == 0] != 0):
            raise ValueError(f"{mod}: missing features must be zero")
        if mod != "lyrics" and np.any(np.linalg.norm(array[mask == 1], axis=1) == 0):
            raise ValueError(f"{mod}: zero feature cannot have a valid mask")


class FeatureTensorDataset(Dataset):
    """Features, labels and IDs always follow the same dataframe rows.

    Missing cache IDs raise by default. Explicit missing_policy='drop' filters
    metadata itself and exposes/warns about dropped IDs. Missing modalities do
    not drop songs: those have zero features and mask=0.
    """
    def __init__(self, df, lyrics_feats, cover_feats, audio_feats,
                 lyrics_masks, cover_masks, audio_masks, song_id_map,
                 missing_policy="error"):
        validate_metadata(df)
        if missing_policy not in ("error", "drop"):
            raise ValueError("missing_policy must be error or drop")
        features = dict(zip(MODALITIES, (lyrics_feats, cover_feats, audio_feats)))
        masks = dict(zip(MODALITIES, (lyrics_masks, cover_masks, audio_masks)))
        validate_arrays(features, masks, song_id_map)
        known = df.song_id.isin(song_id_map)
        self.dropped_song_ids = df.loc[~known, "song_id"].tolist()
        if self.dropped_song_ids:
            if missing_policy == "error":
                raise ValueError(f"Missing cache song IDs: {self.dropped_song_ids[:10]}")
            warnings.warn(f"Explicitly dropped {len(self.dropped_song_ids)} metadata rows absent from cache", UserWarning)
        self.df = df.loc[known].copy().reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No aligned samples remain")
        self.song_ids = self.df.song_id.tolist()
        self.indices = np.asarray([song_id_map[sid] for sid in self.song_ids], dtype=np.int64)
        for mod in MODALITIES:
            setattr(self, f"{mod}_feats", torch.tensor(features[mod][self.indices], dtype=torch.float32))
            setattr(self, f"{mod}_masks", torch.tensor(masks[mod][self.indices], dtype=torch.float32))
        self.labels = torch.tensor([GENRE_TO_IDX[g] for g in self.df.genre], dtype=torch.long)

    @classmethod
    def from_bundle(cls, df, bundle, **kwargs):
        return cls(df, *(bundle.features[m] for m in MODALITIES),
                   *(bundle.masks[m] for m in MODALITIES), bundle.song_id_map, **kwargs)

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        sample = {f"{m}_feat": getattr(self, f"{m}_feats")[idx] for m in MODALITIES}
        sample.update({f"has_{m}": getattr(self, f"{m}_masks")[idx] for m in MODALITIES})
        sample.update(label=self.labels[idx], song_id=self.song_ids[idx])
        return sample


class RMVMusicDataset(FeatureTensorDataset):
    """Legacy public name, now requiring an explicit physical-feature cache."""
    def __init__(self, metadata_path, audio_dim=128, lyrics_dim=5000, cover_dim=512,
                 is_train=False, *, cache_dir=None, feature_bundle=None):
        if (lyrics_dim, cover_dim, audio_dim) != (5000, 512, 128):
            raise ValueError("Official feature dimensions are lyrics=5000, cover=512, audio=128")
        if feature_bundle is None:
            if cache_dir is None:
                raise ValueError("An explicit physical feature cache is required; zero placeholders were removed")
            from .cache import load_feature_cache
            feature_bundle = load_feature_cache(cache_dir)
        df = read_metadata(metadata_path) if isinstance(metadata_path, (str, Path)) else metadata_path
        self.is_train = is_train
        super().__init__(df, *(feature_bundle.features[m] for m in MODALITIES),
                         *(feature_bundle.masks[m] for m in MODALITIES), feature_bundle.song_id_map)


def create_dataloader(csv_path, batch_size=32, shuffle=False, num_workers=0, **dataset_kwargs):
    return DataLoader(RMVMusicDataset(csv_path, **dataset_kwargs), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
