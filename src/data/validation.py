"""Shared metadata and split contracts. Invalid rows are never silently dropped."""
import unicodedata
import itertools

import pandas as pd
import numpy as np

from ..config import GENRE_CLASSES


def normalize_artist(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artist must be a nonempty string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_metadata(df, all_classes=False):
    required = {"song_id", "genre"}
    if not required <= set(df.columns) or len(df) == 0:
        raise ValueError("Metadata must contain nonempty song_id and genre columns")
    if any(not isinstance(x, str) or not x or x != x.strip() for x in df.song_id):
        raise ValueError("song_id must be a nonempty string without outer whitespace")
    if df.song_id.duplicated().any():
        raise ValueError("Duplicate song_id in metadata")
    if not set(df.genre) <= set(GENRE_CLASSES):
        raise ValueError(f"Unknown genre: {set(df.genre) - set(GENRE_CLASSES)}")
    if all_classes and set(df.genre) != set(GENRE_CLASSES):
        raise ValueError("Official metadata must contain all 12 classes")
    return df


def read_metadata(path, all_classes=False):
    df = pd.read_csv(path, keep_default_na=False, dtype={"song_id": str})
    return validate_metadata(df, all_classes=all_classes)


def validate_splits(metadata, partitions, scenario):
    if set(partitions) != {"train", "val", "test"}:
        raise ValueError("Exactly train, val, test are required")
    validate_metadata(metadata)
    indexed = metadata.set_index("song_id")
    for name, df in partitions.items():
        validate_metadata(df)
        if not set(df.song_id) <= set(metadata.song_id):
            raise ValueError(f"{name} has unknown song IDs")
        if set(df.columns) != set(metadata.columns):
            raise ValueError(f"{name} metadata schema differs from source")
        expected = indexed.loc[df.song_id].reset_index()[metadata.columns].reset_index(drop=True)
        actual = df[metadata.columns].reset_index(drop=True)
        # CSV round-trips may infer numeric types differently; compare normalized strings.
        for col in metadata.columns:
            a, b = actual[col], expected[col]
            if not a.astype(str).equals(b.astype(str)):
                if not (pd.to_numeric(a, errors="coerce").notna().all() and
                        np.array_equal(pd.to_numeric(a, errors="coerce").to_numpy(), pd.to_numeric(b, errors="coerce").to_numpy())):
                    raise ValueError(f"{name} metadata mismatch in {col}")
    for a, b in itertools.combinations(partitions, 2):
        if set(partitions[a].song_id) & set(partitions[b].song_id):
            raise ValueError(f"Song leakage: {a}/{b}")
        if scenario == "artist_disjoint":
            if set(partitions[a].artist.map(normalize_artist)) & set(partitions[b].artist.map(normalize_artist)):
                raise ValueError(f"Artist leakage: {a}/{b}")
    expected_ids = set(metadata.song_id)
    if scenario == "temporal":
        if not {"year_status", "release_year"} <= set(metadata.columns):
            raise ValueError("Temporal evaluation requires verified release years")
        years = pd.to_numeric(metadata.release_year, errors="coerce")
        valid = metadata.year_status.eq("verified") & years.notna()
        expected_ids = set(metadata.loc[valid, "song_id"])
        for name, df in partitions.items():
            y = pd.to_numeric(df.release_year, errors="coerce")
            valid_years = y.le(2018) if name == "train" else y.between(2019, 2020) if name == "val" else y.ge(2021)
            if not (df.year_status.eq("verified") & valid_years).all():
                raise ValueError(f"Temporal boundary/verification violation: {name}")
    covered = set().union(*(set(d.song_id) for d in partitions.values()))
    if covered != expected_ids:
        raise ValueError("Split coverage differs from expected metadata subset")
    return {name: {"rows": len(df), "classes": df.genre.value_counts().to_dict()}
            for name, df in partitions.items()}
