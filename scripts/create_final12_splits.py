"""
create_final12_splits.py
RM-VMusic Phase 7B: Benchmark Splits Generator for Final 12-Class Dataset.

Outputs:
- data/splits/final12_iid_train.csv, final12_iid_val.csv, final12_iid_test.csv
- data/splits/final12_artist_disjoint_train.csv, final12_artist_disjoint_val.csv, final12_artist_disjoint_test.csv
- data/splits/final12_temporal_train.csv, final12_temporal_val.csv, final12_temporal_test.csv
- data/splits/final12_label_shift_train.csv, final12_label_shift_val.csv, final12_label_shift_test.csv
- data/splits/final12_missing_modality.csv
"""

import sys
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CSV = BASE_DIR / "data" / "processed" / "final_12class_metadata.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def create_all_12class_splits():
    set_seed(SEED)
    print("=== RM-VMusic Phase 7B: Generating Final 12-Class Benchmark Splits ===")
    
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded {len(df):,} tracks from {DATASET_CSV}")
    
    # -------------------------------------------------------------
    # 1. Standard Stratified IID Split (70 / 15 / 15)
    # -------------------------------------------------------------
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["genre"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["genre"]
    )
    
    train_df = train_df.sort_values(by="song_id").reset_index(drop=True)
    val_df = val_df.sort_values(by="song_id").reset_index(drop=True)
    test_df = test_df.sort_values(by="song_id").reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    train_df.to_csv(SPLITS_DIR / "final12_iid_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(SPLITS_DIR / "final12_iid_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(SPLITS_DIR / "final12_iid_test.csv", index=False, encoding="utf-8")
    print(f"1. IID Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    # -------------------------------------------------------------
    # 2. Strict 0% Artist-Disjoint Split (70 / 15 / 15)
    # -------------------------------------------------------------
    # Group songs by artist
    artist_groups = {}
    for artist, group in df.groupby("artist"):
        artist_groups[artist] = group
        
    artists = list(artist_groups.keys())
    random.Random(SEED).shuffle(artists)
    
    # Stratified-aware artist bin packing
    target_train_size = int(0.70 * len(df))
    target_val_size = int(0.15 * len(df))
    
    train_artists = []
    val_artists = []
    test_artists = []
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for art in artists:
        art_size = len(artist_groups[art])
        if train_count + art_size <= target_train_size:
            train_artists.append(art)
            train_count += art_size
        elif val_count + art_size <= target_val_size:
            val_artists.append(art)
            val_count += art_size
        else:
            test_artists.append(art)
            test_count += art_size
            
    train_ad_df = pd.concat([artist_groups[a] for a in train_artists]).sort_values(by="song_id").reset_index(drop=True)
    val_ad_df = pd.concat([artist_groups[a] for a in val_artists]).sort_values(by="song_id").reset_index(drop=True)
    test_ad_df = pd.concat([artist_groups[a] for a in test_artists]).sort_values(by="song_id").reset_index(drop=True)
    
    train_ad_df["split"] = "train"
    val_ad_df["split"] = "val"
    test_ad_df["split"] = "test"
    
    # Verify strict 0% artist overlap
    s_tr = set(train_ad_df["artist"])
    s_va = set(val_ad_df["artist"])
    s_te = set(test_ad_df["artist"])
    assert len(s_tr & s_va) == 0, "Artist leakage in Train ∩ Val"
    assert len(s_tr & s_te) == 0, "Artist leakage in Train ∩ Test"
    assert len(s_va & s_te) == 0, "Artist leakage in Val ∩ Test"
    
    train_ad_df.to_csv(SPLITS_DIR / "final12_artist_disjoint_train.csv", index=False, encoding="utf-8")
    val_ad_df.to_csv(SPLITS_DIR / "final12_artist_disjoint_val.csv", index=False, encoding="utf-8")
    test_ad_df.to_csv(SPLITS_DIR / "final12_artist_disjoint_test.csv", index=False, encoding="utf-8")
    print(f"2. Artist Disjoint Split: Train={len(train_ad_df):,} (Artists: {len(s_tr):,}), Val={len(val_ad_df):,} (Artists: {len(s_va):,}), Test={len(test_ad_df):,} (Artists: {len(s_te):,}) [0% LEAKAGE]")

    # -------------------------------------------------------------
    # 3. Temporal Shift Split (Strict Verified Years)
    # -------------------------------------------------------------
    # Train <= 2018 | Val = 2019-2020 | Test >= 2021
    df_temp = df[df["year_status"] == "verified"].copy()
    df_temp["year_num"] = pd.to_numeric(df_temp["release_year"], errors="coerce")
    df_temp = df_temp.dropna(subset=["year_num"])
    df_temp["year_num"] = df_temp["year_num"].astype(int)
    
    temp_train = df_temp[df_temp["year_num"] <= 2018].sort_values(by="song_id").reset_index(drop=True)
    temp_val = df_temp[(df_temp["year_num"] >= 2019) & (df_temp["year_num"] <= 2020)].sort_values(by="song_id").reset_index(drop=True)
    temp_test = df_temp[df_temp["year_num"] >= 2021].sort_values(by="song_id").reset_index(drop=True)
    
    temp_train["split"] = "train"
    temp_val["split"] = "val"
    temp_test["split"] = "test"
    
    temp_train.to_csv(SPLITS_DIR / "final12_temporal_train.csv", index=False, encoding="utf-8")
    temp_val.to_csv(SPLITS_DIR / "final12_temporal_val.csv", index=False, encoding="utf-8")
    temp_test.to_csv(SPLITS_DIR / "final12_temporal_test.csv", index=False, encoding="utf-8")
    print(f"3. Temporal Shift: Total Verified={len(df_temp):,} (Train={len(temp_train):,}, Val={len(temp_val):,}, Test={len(temp_test):,}) | Excluded unverified: {len(df) - len(df_temp):,}")

    # -------------------------------------------------------------
    # 4. Controlled Label Shift Split
    # -------------------------------------------------------------
    # Skew test distribution by changing class priors
    train_ls_rows = []
    val_ls_rows = []
    test_ls_rows = []
    
    for g, group in df.groupby("genre"):
        g_shuffled = group.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(g_shuffled)
        
        # Define prior shifting weights
        if g in ["POP_BALLAD", "BOLERO_TRUTINH"]:
            # Reduce dominance in test (shift toward minority)
            tr_cut = int(0.78 * n)
            va_cut = int(0.88 * n)
        else:
            # Increase representation in test
            tr_cut = int(0.60 * n)
            va_cut = int(0.72 * n)
            
        train_ls_rows.append(g_shuffled.iloc[:tr_cut])
        val_ls_rows.append(g_shuffled.iloc[tr_cut:va_cut])
        test_ls_rows.append(g_shuffled.iloc[va_cut:])
        
    train_ls_df = pd.concat(train_ls_rows).sort_values(by="song_id").reset_index(drop=True)
    val_ls_df = pd.concat(val_ls_rows).sort_values(by="song_id").reset_index(drop=True)
    test_ls_df = pd.concat(test_ls_rows).sort_values(by="song_id").reset_index(drop=True)
    
    train_ls_df["split"] = "train"
    val_ls_df["split"] = "val"
    test_ls_df["split"] = "test"
    
    train_ls_df.to_csv(SPLITS_DIR / "final12_label_shift_train.csv", index=False, encoding="utf-8")
    val_ls_df.to_csv(SPLITS_DIR / "final12_label_shift_val.csv", index=False, encoding="utf-8")
    test_ls_df.to_csv(SPLITS_DIR / "final12_label_shift_test.csv", index=False, encoding="utf-8")
    print(f"4. Label Shift: Train={len(train_ls_df):,}, Val={len(val_ls_df):,}, Test={len(test_ls_df):,}")

    # -------------------------------------------------------------
    # 5. Missing Modality Split
    # -------------------------------------------------------------
    df_missing = df.copy()
    df_missing.to_csv(SPLITS_DIR / "final12_missing_modality.csv", index=False, encoding="utf-8")
    print(f"5. Missing Modality Dataset written ({len(df_missing):,} records)")

if __name__ == "__main__":
    create_all_12class_splits()
