"""
Temporal Split Module.
Splits by verified publication year into historical, transition, and modern partitions.
"""
import pandas as pd

def split_temporal(df, year_col="release_year", status_col="year_status", train_max_year=2018, val_min_year=2019, val_max_year=2020, test_min_year=2021):
    """
    Generates chronological partitions on verified release year records:
    - Train: <= 2018 (Historical)
    - Val: 2019-2020 (Transition)
    - Test: >= 2021 (Modern)
    """
    if status_col in df.columns:
        df_valid = df[df[status_col] == "verified"].copy()
    else:
        df_valid = df[df[year_col].notna()].copy()
        
    df_valid[year_col] = pd.to_numeric(df_valid[year_col], errors="coerce")
    df_valid = df_valid.dropna(subset=[year_col])
    
    tr_df = df_valid[df_valid[year_col] <= train_max_year].copy()
    va_df = df_valid[(df_valid[year_col] >= val_min_year) & (df_valid[year_col] <= val_max_year)].copy()
    te_df = df_valid[df_valid[year_col] >= test_min_year].copy()
    
    return tr_df, va_df, te_df
