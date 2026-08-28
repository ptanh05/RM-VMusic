"""
Artist-Disjoint Split Module.
Ensures mathematically proven 0% artist leakage between Train, Val, and Test.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from .base import verify_zero_artist_leakage

def split_artist_disjoint(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42, artist_col="artist", genre_col="genre"):
    """
    Performs group stratified artist-disjoint splitting.
    """
    # 1. Aggregate primary genre per artist for balanced stratification
    artist_meta = df.groupby(artist_col).agg(
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
    
    tr_art_set = set(tr_artists[artist_col])
    va_art_set = set(va_artists[artist_col])
    te_art_set = set(te_artists[artist_col])
    
    tr_df = df[df[artist_col].isin(tr_art_set)].copy()
    va_df = df[df[artist_col].isin(va_art_set)].copy()
    te_df = df[df[artist_col].isin(te_art_set)].copy()
    
    # Verify zero leakage
    verify_zero_artist_leakage(tr_df, va_df, te_df, artist_col=artist_col)
    
    return tr_df, va_df, te_df
