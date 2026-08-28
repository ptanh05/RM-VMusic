"""
Base verification and integrity checks for RM-VMusic splitters.
"""
import pandas as pd

def verify_zero_artist_leakage(tr_df, va_df, te_df, artist_col="artist"):
    """
    Mathematically verifies that Train, Val, and Test artist sets are mutually disjoint.
    """
    tr_art = set(tr_df[artist_col].dropna())
    va_art = set(va_df[artist_col].dropna())
    te_art = set(te_df[artist_col].dropna())
    
    l_tr_va = tr_art & va_art
    l_tr_te = tr_art & te_art
    l_va_te = va_art & te_art
    
    total_leakage = len(l_tr_va) + len(l_tr_te) + len(l_va_te)
    if total_leakage > 0:
        raise ValueError(
            f"ARTIST LEAKAGE DETECTED: Tr∩Va={len(l_tr_va)}, Tr∩Te={len(l_tr_te)}, Va∩Te={len(l_va_te)}"
        )
    return True

def verify_12_classes(df, genre_col="genre", expected_count=12):
    """
    Verifies that all 12 classes exist in the target partition.
    """
    n = df[genre_col].nunique()
    if n != expected_count:
        raise ValueError(f"Expected {expected_count} classes, found {n}")
    return True
