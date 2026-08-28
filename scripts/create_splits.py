"""
create_splits.py
RM-VMusic: Distribution Shift Benchmark Split Generator on Core Trainable Dataset.
1. iid.csv (Stratified IID Benchmark Split: 70% Train, 15% Val, 15% Test)
2. artist_disjoint.csv (Strict 0.00% Artist Leakage Group Split)
3. temporal.csv (Strict Verified Release Year Partition: Train <= 2018, Val 2019-2020, Test >= 2021)
4. missing_modality.csv (Complete Multimodal Train vs Incomplete Eval)
5. label_shift.csv (Prior Genre Distribution Shift)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TRAINABLE_METADATA_PATH = PROCESSED_DIR / "trainable_metadata.csv"

def create_iid_split(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["split"] = "unassigned"
    np.random.seed(42)
    
    for _, group in df_out.groupby("genre"):
        indices = group.index.tolist()
        np.random.shuffle(indices)
        n = len(indices)
        
        if n == 1:
            df_out.loc[indices[0], "split"] = "train"
        elif n == 2:
            df_out.loc[indices[0], "split"] = "train"
            df_out.loc[indices[1], "split"] = "test"
        elif n == 3:
            df_out.loc[indices[0], "split"] = "train"
            df_out.loc[indices[1], "split"] = "val"
            df_out.loc[indices[2], "split"] = "test"
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.15)))
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            df_out.loc[train_idx, "split"] = "train"
            df_out.loc[val_idx, "split"] = "val"
            df_out.loc[test_idx, "split"] = "test"
            
    return df_out

def create_artist_disjoint_split(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    artists = list(df_out["artist_id"].unique())
    np.random.seed(42)
    np.random.shuffle(artists)
    
    n_artists = len(artists)
    n_train = int(n_artists * 0.70)
    n_val = int(n_artists * 0.15)
    
    train_artists = set(artists[:n_train])
    val_artists = set(artists[n_train:n_train + n_val])
    test_artists = set(artists[n_train + n_val:])
    
    def assign_split(aid):
        if aid in train_artists:
            return "train"
        elif aid in val_artists:
            return "val"
        else:
            return "test"
            
    df_out["split"] = df_out["artist_id"].apply(assign_split)
    return df_out

def create_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates temporal shift ONLY on records with verified release_year.
    Unverified records are marked 'UNVERIFIED_YEAR' and excluded from evaluation sets.
    """
    df_out = df.copy()
    df_out["split"] = "UNVERIFIED_YEAR"
    
    verified_mask = df_out["release_year"].notna() & (df_out["release_year_source"] != "unverified_null")
    print(f"\nTemporal Split: {verified_mask.sum()} samples have verified release years.")
    
    for idx in df_out[verified_mask].index:
        yr = int(df_out.loc[idx, "release_year"])
        if yr <= 2018:
            df_out.loc[idx, "split"] = "train"
        elif 2019 <= yr <= 2020:
            df_out.loc[idx, "split"] = "val"
        else:
            df_out.loc[idx, "split"] = "test"
            
    return df_out

def create_missing_modality_split(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    has_lyrics = df_out["lyrics"].notna() & (df_out["lyrics"] != "")
    has_audio = df_out["audio_url"].notna() & (df_out["audio_url"] != "")
    has_cover = df_out["cover_url"].notna() & (df_out["cover_url"] != "")
    
    complete_mm = has_lyrics & has_audio & has_cover
    df_out["split"] = "train"
    
    mm_indices = df_out[complete_mm].index.tolist()
    np.random.seed(42)
    np.random.shuffle(mm_indices)
    
    n_mm = len(mm_indices)
    n_train = int(n_mm * 0.70)
    n_val = int(n_mm * 0.15)
    
    df_out.loc[mm_indices[:n_train], "split"] = "train"
    df_out.loc[mm_indices[n_train:n_train + n_val], "split"] = "val"
    df_out.loc[mm_indices[n_train + n_val:], "split"] = "test"
    
    inc_indices = df_out[~complete_mm].index.tolist()
    for pos, i_idx in enumerate(inc_indices):
        df_out.loc[i_idx, "split"] = "val" if pos % 2 == 0 else "test"
        
    return df_out

def create_label_shift_split(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    dominant_genres = ["POP_BALLAD", "BOLERO_TRUTINH"]
    minority_genres = [
        "RAP_HIPHOP", "ROCK", "DANCE_EDM", "REVOLUTIONARY",
        "FOLK_TRADITIONAL", "NHAC_TRINH", "RB_SOUL", "CHILDREN", "INSTRUMENTAL"
    ]
    df_out["split"] = "unassigned"
    np.random.seed(42)
    
    dom_idx = df_out[df_out["genre"].isin(dominant_genres)].index.tolist()
    np.random.shuffle(dom_idx)
    n_dom = len(dom_idx)
    n_dom_train = int(n_dom * 0.80)
    n_dom_val = int(n_dom * 0.10)
    
    df_out.loc[dom_idx[:n_dom_train], "split"] = "train"
    df_out.loc[dom_idx[n_dom_train:n_dom_train + n_dom_val], "split"] = "val"
    df_out.loc[dom_idx[n_dom_train + n_dom_val:], "split"] = "test"
    
    min_idx = df_out[df_out["genre"].isin(minority_genres)].index.tolist()
    np.random.shuffle(min_idx)
    n_min = len(min_idx)
    n_min_train = int(n_min * 0.40)
    n_min_val = int(n_min * 0.20)
    
    df_out.loc[min_idx[:n_min_train], "split"] = "train"
    df_out.loc[min_idx[n_min_train:n_min_train + n_min_val], "split"] = "val"
    df_out.loc[min_idx[n_min_train + n_min_val:], "split"] = "test"
    
    return df_out

def main():
    print("=== RM-VMusic: Generating Distribution Shift Benchmark Splits ===")
    if not TRAINABLE_METADATA_PATH.exists():
        print(f"[ERROR] Trainable metadata not found at {TRAINABLE_METADATA_PATH}.")
        sys.exit(1)
        
    df_trainable = pd.read_csv(TRAINABLE_METADATA_PATH)
    print(f"Loaded Trainable Dataset: {len(df_trainable)} samples across {df_trainable['genre'].nunique()} genres.")
    
    # 1. IID Split
    df_iid = create_iid_split(df_trainable)
    df_iid.to_csv(SPLITS_DIR / "iid.csv", index=False, encoding="utf-8")
    print(f"[OK] IID Split: Train={ (df_iid['split']=='train').sum()}, Val={(df_iid['split']=='val').sum()}, Test={(df_iid['split']=='test').sum()}")
    
    # 2. Artist-Disjoint Split
    df_artist = create_artist_disjoint_split(df_trainable)
    df_artist.to_csv(SPLITS_DIR / "artist_disjoint.csv", index=False, encoding="utf-8")
    print(f"[OK] Artist-Disjoint Split: Train={(df_artist['split']=='train').sum()}, Val={(df_artist['split']=='val').sum()}, Test={(df_artist['split']=='test').sum()}")
    
    # 3. Temporal Split
    df_temporal = create_temporal_split(df_trainable)
    df_temporal.to_csv(SPLITS_DIR / "temporal.csv", index=False, encoding="utf-8")
    print(f"[OK] Temporal Split Breakdown: {df_temporal['split'].value_counts().to_dict()}")
    
    # 4. Missing Modality Split
    df_missing = create_missing_modality_split(df_trainable)
    df_missing.to_csv(SPLITS_DIR / "missing_modality.csv", index=False, encoding="utf-8")
    print(f"[OK] Missing Modality Split: Train={(df_missing['split']=='train').sum()}, Val={(df_missing['split']=='val').sum()}, Test={(df_missing['split']=='test').sum()}")
    
    # 5. Label Shift Split
    df_label = create_label_shift_split(df_trainable)
    df_label.to_csv(SPLITS_DIR / "label_shift.csv", index=False, encoding="utf-8")
    print(f"[OK] Label Shift Split: Train={(df_label['split']=='train').sum()}, Val={(df_label['split']=='val').sum()}, Test={(df_label['split']=='test').sum()}")
    
    print("=== Split Generation Completed ===")

if __name__ == "__main__":
    main()
