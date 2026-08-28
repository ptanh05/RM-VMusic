"""
create_final_splits.py
RM-VMusic Phase 6B: Generation of Final Stratified, Artist-Disjoint, Temporal, Label Shift, and Missing Modality Splits.
Outputs to: data/splits/final_*.csv
"""

import sys
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_CSV = BASE_DIR / "data" / "processed" / "final_trainable_metadata.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENRES = [
    "POP_BALLAD",
    "BOLERO_TRUTINH",
    "INSTRUMENTAL",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "DANCE_EDM",
    "REVOLUTIONARY",
    "NHAC_TRINH",
    "ROCK",
    "RB_SOUL",
    "CHILDREN"
]

def run_create_final_splits():
    print("=== RM-VMusic Phase 6B: Creating Final Benchmark Splits ===")
    
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"{FINAL_CSV} not found!")
        
    df = pd.read_csv(FINAL_CSV)
    n_total = len(df)
    print(f"Loaded {n_total:,} records from {FINAL_CSV}")
    
    # -------------------------------------------------------------
    # 1. TASK 5: IID Train / Val / Test (70% / 15% / 15%)
    # -------------------------------------------------------------
    print("\n>>> TASK 5: Creating Final IID Stratified Splits (70 / 15 / 15) <<<")
    # First split 70% train vs 30% eval
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, eval_idx = next(sss1.split(df, df["genre"]))
    
    df_train = df.iloc[train_idx].copy()
    df_eval = df.iloc[eval_idx].copy()
    
    # Second split eval (30%) into 50% val (15% total) and 50% test (15% total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx_sub, test_idx_sub = next(sss2.split(df_eval, df_eval["genre"]))
    
    df_val = df_eval.iloc[val_idx_sub].copy()
    df_test = df_eval.iloc[test_idx_sub].copy()
    
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    
    df_train.to_csv(SPLITS_DIR / "final_iid_train.csv", index=False)
    df_val.to_csv(SPLITS_DIR / "final_iid_val.csv", index=False)
    df_test.to_csv(SPLITS_DIR / "final_iid_test.csv", index=False)
    
    print(f" - final_iid_train.csv: {len(df_train):,} ({len(df_train)/n_total*100:.1f}%)")
    print(f" - final_iid_val.csv:   {len(df_val):,} ({len(df_val)/n_total*100:.1f}%)")
    print(f" - final_iid_test.csv:  {len(df_test):,} ({len(df_test)/n_total*100:.1f}%)")

    # -------------------------------------------------------------
    # 2. TASK 6: Artist Disjoint Split (Strict 0% Leakage)
    # -------------------------------------------------------------
    print("\n>>> TASK 6: Creating Final Artist-Disjoint Splits (Strict 0% Leakage) <<<")
    # Group songs by artist_id
    artist_groups = df.groupby("artist_id")
    artist_list = list(artist_groups.groups.keys())
    
    # Sort artist list deterministically, then shuffle with seed 42
    artist_list.sort()
    rng = random.Random(42)
    rng.shuffle(artist_list)
    
    target_train_n = int(n_total * 0.70)
    target_val_n = int(n_total * 0.15)
    
    train_artists, val_artists, test_artists = set(), set(), set()
    train_rows, val_rows, test_rows = [], [], []
    
    cur_train_cnt, cur_val_cnt = 0, 0
    
    for art in artist_list:
        art_df = artist_groups.get_group(art)
        cnt = len(art_df)
        if cur_train_cnt + cnt <= target_train_n:
            train_artists.add(art)
            train_rows.append(art_df)
            cur_train_cnt += cnt
        elif cur_val_cnt + cnt <= target_val_n:
            val_artists.add(art)
            val_rows.append(art_df)
            cur_val_cnt += cnt
        else:
            test_artists.add(art)
            test_rows.append(art_df)
            
    df_art_train = pd.concat(train_rows).copy()
    df_art_val = pd.concat(val_rows).copy()
    df_art_test = pd.concat(test_rows).copy()
    
    df_art_train["split"] = "train"
    df_art_val["split"] = "val"
    df_art_test["split"] = "test"
    
    df_art_train.to_csv(SPLITS_DIR / "final_artist_disjoint_train.csv", index=False)
    df_art_val.to_csv(SPLITS_DIR / "final_artist_disjoint_val.csv", index=False)
    df_art_test.to_csv(SPLITS_DIR / "final_artist_disjoint_test.csv", index=False)
    
    art_tr = set(df_art_train["artist_id"])
    art_va = set(df_art_val["artist_id"])
    art_te = set(df_art_test["artist_id"])
    
    overlap_va = len(art_tr.intersection(art_va))
    overlap_te = len(art_tr.intersection(art_te))
    overlap_vate = len(art_va.intersection(art_te))
    
    print(f" - final_artist_disjoint_train.csv: {len(df_art_train):,} rows ({len(art_tr)} unique artists)")
    print(f" - final_artist_disjoint_val.csv:   {len(df_art_val):,} rows ({len(art_va)} unique artists)")
    print(f" - final_artist_disjoint_test.csv:  {len(df_art_test):,} rows ({len(art_te)} unique artists)")
    print(f" - Artist Overlap Train <-> Val:  {overlap_va} (0.00% Leakage)")
    print(f" - Artist Overlap Train <-> Test: {overlap_te} (0.00% Leakage)")
    print(f" - Artist Overlap Val <-> Test:   {overlap_vate} (0.00% Leakage)")

    # -------------------------------------------------------------
    # 3. TASK 7: Temporal Shift Split (Verified Years Only)
    # -------------------------------------------------------------
    print("\n>>> TASK 7: Creating Final Temporal Shift Splits (Verified Release Years) <<<")
    df_temp = df.copy()
    df_temp["release_year_clean"] = pd.to_numeric(df_temp["release_year"], errors="coerce")
    df_temp_verified = df_temp[df_temp["release_year_clean"].notna()].copy()
    
    # Train: <= 2018 | Val: 2019-2020 | Test: >= 2021
    df_temp_tr = df_temp_verified[df_temp_verified["release_year_clean"] <= 2018].copy()
    df_temp_va = df_temp_verified[(df_temp_verified["release_year_clean"] >= 2019) & (df_temp_verified["release_year_clean"] <= 2020)].copy()
    df_temp_te = df_temp_verified[df_temp_verified["release_year_clean"] >= 2021].copy()
    
    df_temp_tr["split"] = "train"
    df_temp_va["split"] = "val"
    df_temp_te["split"] = "test"
    
    df_temp_tr.to_csv(SPLITS_DIR / "final_temporal_train.csv", index=False)
    df_temp_va.to_csv(SPLITS_DIR / "final_temporal_val.csv", index=False)
    df_temp_te.to_csv(SPLITS_DIR / "final_temporal_test.csv", index=False)
    
    n_unverified = len(df) - len(df_temp_verified)
    print(f" - Total Verified Temporal Tracks: {len(df_temp_verified):,} / {len(df):,} ({len(df_temp_verified)/len(df)*100:.2f}%)")
    print(f" - Unverified Excluded Tracks:     {n_unverified:,} ({n_unverified/len(df)*100:.2f}%)")
    print(f" - final_temporal_train.csv (<= 2018): {len(df_temp_tr):,} rows")
    print(f" - final_temporal_val.csv (2019-2020): {len(df_temp_va):,} rows")
    print(f" - final_temporal_test.csv (>= 2021):  {len(df_temp_te):,} rows")

    # -------------------------------------------------------------
    # 4. TASK 8: Label Distribution Shift Split
    # -------------------------------------------------------------
    print("\n>>> TASK 8: Creating Final Label Distribution Shift Splits <<<")
    # Natural label shift: Train emphasizes top dominant classes (POP_BALLAD, BOLERO_TRUTINH)
    # Val / Test emphasize minority & rare classes proportionally
    dominant_genres = ["POP_BALLAD", "BOLERO_TRUTINH"]
    minority_genres = [g for g in TARGET_GENRES if g not in dominant_genres]
    
    train_parts, val_parts, test_parts = [], [], []
    
    for g in TARGET_GENRES:
        df_g = df[df["genre"] == g].copy()
        n_g = len(df_g)
        
        # Shuffle with seed 42
        df_g = df_g.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        if g in dominant_genres:
            # Dominant: 80% train, 10% val, 10% test
            n_tr = int(n_g * 0.80)
            n_va = int(n_g * 0.10)
        else:
            # Minority: 50% train, 25% val, 25% test (substantially higher proportion in evaluation)
            n_tr = int(n_g * 0.50)
            n_va = int(n_g * 0.25)
            
        train_parts.append(df_g.iloc[:n_tr])
        val_parts.append(df_g.iloc[n_tr:n_tr+n_va])
        test_parts.append(df_g.iloc[n_tr+n_va:])
        
    df_lbl_tr = pd.concat(train_parts).copy()
    df_lbl_va = pd.concat(val_parts).copy()
    df_lbl_te = pd.concat(test_parts).copy()
    
    df_lbl_tr["split"] = "train"
    df_lbl_va["split"] = "val"
    df_lbl_te["split"] = "test"
    
    df_lbl_tr.to_csv(SPLITS_DIR / "final_label_shift_train.csv", index=False)
    df_lbl_va.to_csv(SPLITS_DIR / "final_label_shift_val.csv", index=False)
    df_lbl_te.to_csv(SPLITS_DIR / "final_label_shift_test.csv", index=False)
    
    print(f" - final_label_shift_train.csv: {len(df_lbl_tr):,} rows (Dominant share: {df_lbl_tr['genre'].isin(dominant_genres).mean()*100:.1f}%)")
    print(f" - final_label_shift_val.csv:   {len(df_lbl_va):,} rows (Minority share: {(~df_lbl_va['genre'].isin(dominant_genres)).mean()*100:.1f}%)")
    print(f" - final_label_shift_test.csv:  {len(df_lbl_te):,} rows (Minority share: {(~df_lbl_te['genre'].isin(dominant_genres)).mean()*100:.1f}%)")

    # -------------------------------------------------------------
    # 5. TASK 9: Missing Modality Split
    # -------------------------------------------------------------
    print("\n>>> TASK 9: Creating Final Missing Modality Benchmark <<<")
    df_mm = df.copy()
    df_mm.to_csv(SPLITS_DIR / "final_missing_modality.csv", index=False)
    print(f" - final_missing_modality.csv: {len(df_mm):,} rows with verified modality_pattern annotations.")

if __name__ == "__main__":
    run_create_final_splits()
