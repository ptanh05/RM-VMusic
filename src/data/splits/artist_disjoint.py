"""
Artist-Disjoint Split Module.
Ensures mathematically proven 0% artist leakage between Train, Val, and Test.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from .base import verify_zero_artist_leakage
from ..validation import normalize_artist

def split_artist_disjoint(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42, artist_col="artist", genre_col="genre"):
    """
    Performs group stratified artist-disjoint splitting.
    """
    df_copy = df.copy()
    norm_col = "_artist_norm"
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9 or min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive and sum to 1")
    df_copy[norm_col] = df_copy[artist_col].map(normalize_artist)

    # 1. Aggregate primary genre per artist for balanced stratification
    artist_meta = df_copy.groupby(norm_col).agg(
        song_count=("song_id", "count"),
        primary_genre=(genre_col, lambda x: x.mode()[0])
    ).reset_index()
    
    test_size = val_ratio + test_ratio
    tr_artists, temp_artists = train_test_split(
        artist_meta, test_size=test_size, random_state=random_state, stratify=artist_meta["primary_genre"]
    )
    val_rel_size = val_ratio / test_size
    va_artists, te_artists = train_test_split(
        temp_artists, test_size=(1.0 - val_rel_size), random_state=random_state, stratify=temp_artists["primary_genre"]
    )
    
    tr_art_set = set(tr_artists[norm_col])
    va_art_set = set(va_artists[norm_col])
    te_art_set = set(te_artists[norm_col])
    
    tr_df = df_copy[df_copy[norm_col].isin(tr_art_set)].drop(columns=[norm_col]).copy()
    va_df = df_copy[df_copy[norm_col].isin(va_art_set)].drop(columns=[norm_col]).copy()
    te_df = df_copy[df_copy[norm_col].isin(te_art_set)].drop(columns=[norm_col]).copy()
    
    # Verify zero leakage
    verify_zero_artist_leakage(tr_df, va_df, te_df, artist_col=artist_col)
    
    return tr_df, va_df, te_df
