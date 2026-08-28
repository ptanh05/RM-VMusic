"""
create_final_12class_splits.py
RM-VMusic Phase 7: Task 12 - 12-Class Benchmark Splits Generation.
Outputs to: data/splits/final_12class_*.csv
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
FINAL_12_CSV = BASE_DIR / "data" / "processed" / "final_12class_metadata.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

GENRES_12 = [
    "POP_BALLAD",
    "BOLERO_TRUTINH",
    "DANCE_EDM",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "CHILDREN",
    "REVOLUTIONARY",
    "RB_SOUL",
    "NHAC_TRINH",
    "INSTRUMENTAL",
    "ROCK",
    "OTHER"
]

def run_create_12class_splits():
    print("=== RM-VMusic Phase 7: Task 12 - Creating 12-Class Benchmark Splits ===")
    
    if not FINAL_12_CSV.exists():
        raise FileNotFoundError(f"{FINAL_12_CSV} not found!")
        
    df = pd.read_csv(FINAL_12_CSV)
    n_total = len(df)
    print(f"Loaded {n_total:,} records (12 classes) from {FINAL_12_CSV}")
    
    # -------------------------------------------------------------
    # 1. 12-Class IID Train / Val / Test (70% / 15% / 15%)
    # -------------------------------------------------------------
    print("\n>>> 1. Creating 12-Class IID Stratified Splits (70 / 15 / 15) <<<")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, eval_idx = next(sss1.split(df, df["genre"]))
    
    df_train = df.iloc[train_idx].copy()
    df_eval = df.iloc[eval_idx].copy()
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx_sub, test_idx_sub = next(sss2.split(df_eval, df_eval["genre"]))
    
    df_val = df_eval.iloc[val_idx_sub].copy()
    df_test = df_eval.iloc[test_idx_sub].copy()
    
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    
    df_train.to_csv(SPLITS_DIR / "final_12class_iid_train.csv", index=False)
    df_val.to_csv(SPLITS_DIR / "final_12class_iid_val.csv", index=False)
    df_test.to_csv(SPLITS_DIR / "final_12class_iid_test.csv", index=False)
    
    print(f" - final_12class_iid_train.csv: {len(df_train):,} ({len(df_train)/n_total*100:.1f}%)")
    print(f" - final_12class_iid_val.csv:   {len(df_val):,} ({len(df_val)/n_total*100:.1f}%)")
    print(f" - final_12class_iid_test.csv:  {len(df_test):,} ({len(df_test)/n_total*100:.1f}%)")

    # -------------------------------------------------------------
    # 2. 12-Class Artist Disjoint Split (Strict 0% Leakage)
    # -------------------------------------------------------------
    print("\n>>> 2. Creating 12-Class Artist-Disjoint Splits (Strict 0% Leakage) <<<")
    artist_groups = df.groupby("artist_id")
    artist_list = list(artist_groups.groups.keys())
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
            
    df_art_tr = pd.concat(train_rows).copy()
    df_art_va = pd.concat(val_rows).copy()
    df_art_te = pd.concat(test_rows).copy()
    
    df_art_tr["split"] = "train"
    df_art_va["split"] = "val"
    df_art_te["split"] = "test"
    
    df_art_tr.to_csv(SPLITS_DIR / "final_12class_artist_disjoint_train.csv", index=False)
    df_art_va.to_csv(SPLITS_DIR / "final_12class_artist_disjoint_val.csv", index=False)
    df_art_te.to_csv(SPLITS_DIR / "final_12class_artist_disjoint_test.csv", index=False)
    
    art_tr_set = set(df_art_tr["artist_id"])
    art_va_set = set(df_art_va["artist_id"])
    art_te_set = set(df_art_te["artist_id"])
    
    leak_va = len(art_tr_set.intersection(art_va_set))
    leak_te = len(art_tr_set.intersection(art_te_set))
    leak_vate = len(art_va_set.intersection(art_te_set))
    
    print(f" - final_12class_artist_disjoint_train.csv: {len(df_art_tr):,} rows ({len(art_tr_set)} artists)")
    print(f" - final_12class_artist_disjoint_val.csv:   {len(df_art_va):,} rows ({len(art_va_set)} artists)")
    print(f" - final_12class_artist_disjoint_test.csv:  {len(df_art_te):,} rows ({len(art_te_set)} artists)")
    print(f" - Leakage Train <-> Val:  {leak_va} (0.00%)")
    print(f" - Leakage Train <-> Test: {leak_te} (0.00%)")
    print(f" - Leakage Val <-> Test:   {leak_vate} (0.00%)")

    # -------------------------------------------------------------
    # 3. 12-Class Temporal Shift Split
    # -------------------------------------------------------------
    print("\n>>> 3. Creating 12-Class Temporal Shift Splits (Verified Release Years) <<<")
    df_temp = df.copy()
    df_temp["year_num"] = pd.to_numeric(df_temp["release_year"], errors="coerce")
    df_temp_v = df_temp[df_temp["year_num"].notna()].copy()
    
    df_tmp_tr = df_temp_v[df_temp_v["year_num"] <= 2018].copy()
    df_tmp_va = df_temp_v[(df_temp_v["year_num"] >= 2019) & (df_temp_v["year_num"] <= 2020)].copy()
    df_tmp_te = df_temp_v[df_temp_v["year_num"] >= 2021].copy()
    
    df_tmp_tr["split"] = "train"
    df_tmp_va["split"] = "val"
    df_tmp_te["split"] = "test"
    
    df_tmp_tr.to_csv(SPLITS_DIR / "final_12class_temporal_train.csv", index=False)
    df_tmp_va.to_csv(SPLITS_DIR / "final_12class_temporal_val.csv", index=False)
    df_tmp_te.to_csv(SPLITS_DIR / "final_12class_temporal_test.csv", index=False)
    
    print(f" - final_12class_temporal_train.csv (<= 2018): {len(df_tmp_tr):,} rows")
    print(f" - final_12class_temporal_val.csv (2019-2020): {len(df_tmp_va):,} rows")
    print(f" - final_12class_temporal_test.csv (>= 2021):  {len(df_tmp_te):,} rows")

    # -------------------------------------------------------------
    # 4. 12-Class Label Shift Split
    # -------------------------------------------------------------
    print("\n>>> 4. Creating 12-Class Label Shift Splits <<<")
    dom_genres = ["POP_BALLAD", "BOLERO_TRUTINH"]
    train_parts, val_parts, test_parts = [], [], []
    
    for g in GENRES_12:
        df_g = df[df["genre"] == g].copy()
        df_g = df_g.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_g = len(df_g)
        
        if g in dom_genres:
            n_tr = int(n_g * 0.80)
            n_va = int(n_g * 0.10)
        else:
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
    
    df_lbl_tr.to_csv(SPLITS_DIR / "final_12class_label_shift_train.csv", index=False)
    df_lbl_va.to_csv(SPLITS_DIR / "final_12class_label_shift_val.csv", index=False)
    df_lbl_te.to_csv(SPLITS_DIR / "final_12class_label_shift_test.csv", index=False)
    
    print(f" - final_12class_label_shift_train.csv: {len(df_lbl_tr):,} rows")
    print(f" - final_12class_label_shift_val.csv:   {len(df_lbl_va):,} rows")
    print(f" - final_12class_label_shift_test.csv:  {len(df_lbl_te):,} rows")

    # -------------------------------------------------------------
    # 5. 12-Class Missing Modality Benchmark
    # -------------------------------------------------------------
    print("\n>>> 5. Creating 12-Class Missing Modality Benchmark <<<")
    df_mm = df.copy()
    df_mm.to_csv(SPLITS_DIR / "final_12class_missing_modality.csv", index=False)
    print(f" - final_12class_missing_modality.csv: {len(df_mm):,} rows saved.")

if __name__ == "__main__":
    run_create_12class_splits()
