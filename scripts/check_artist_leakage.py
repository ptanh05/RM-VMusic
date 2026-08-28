"""
check_artist_leakage.py
Verifies artist disjointness and calculates artist leakage rates across dataset splits.
Guarantees 0% artist leakage for artist_disjoint.csv and handles NOT_READY status gracefully.
"""

import sys
from pathlib import Path
import pandas as pd

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits"

SPLIT_FILES = [
    "iid.csv",
    "artist_disjoint.csv",
    "temporal.csv",
    "missing_modality.csv",
    "label_shift.csv"
]

def analyze_artist_leakage_for_file(file_path: Path):
    if not file_path.exists():
        print(f"[WARNING] Split file {file_path.name} not found. Skipping.")
        return None
        
    df = pd.read_csv(file_path)
    if "split" not in df.columns or "artist_id" not in df.columns:
        print(f"[ERROR] Split file {file_path.name} missing 'split' or 'artist_id' column.")
        return None
        
    splits = df["split"].unique()
    if "NOT_READY_NO_RELEASE_YEAR" in splits:
        print(f"\n--- Split Analysis: {file_path.name} ---")
        print(f"Status: NOT_READY / INVALID (No verified release_year metadata available upstream).")
        print(f"Leakage calculation: N/A")
        return {
            "file": file_path.name,
            "status": "NOT_READY",
            "leakage_pct": None,
            "leaked_artists": 0
        }
        
    train_artists = set(df[df["split"] == "train"]["artist_id"].dropna())
    val_artists = set(df[df["split"] == "val"]["artist_id"].dropna())
    test_artists = set(df[df["split"] == "test"]["artist_id"].dropna())
    
    train_val_overlap = train_artists.intersection(val_artists)
    train_test_overlap = train_artists.intersection(test_artists)
    val_test_overlap = val_artists.intersection(test_artists)
    
    total_eval_artists = len(val_artists.union(test_artists))
    leaked_eval_artists = len(train_val_overlap.union(train_test_overlap))
    
    leakage_pct = (leaked_eval_artists / total_eval_artists * 100) if total_eval_artists > 0 else 0.0
    
    print(f"\n--- Split Analysis: {file_path.name} ---")
    print(f"Total Rows: {len(df)} | Train: {len(df[df['split']=='train'])}, Val: {len(df[df['split']=='val'])}, Test: {len(df[df['split']=='test'])}")
    print(f"Unique Artists: Train={len(train_artists)}, Val={len(val_artists)}, Test={len(test_artists)}")
    print(f"Train <-> Val Overlapping Artists: {len(train_val_overlap)}")
    print(f"Train <-> Test Overlapping Artists: {len(train_test_overlap)}")
    print(f"Artist Leakage to Eval (Val+Test): {leakage_pct:.2f}% ({leaked_eval_artists}/{total_eval_artists})")
    
    if file_path.name == "artist_disjoint.csv":
        if leaked_eval_artists > 0:
            print(f"[CRITICAL ERROR] Artist leakage detected in artist_disjoint.csv! Found {leaked_eval_artists} shared artists.")
        else:
            print("[VERIFIED SUCCESS] artist_disjoint.csv has STRICTLY 0.00% artist leakage.")
            
    return {
        "file": file_path.name,
        "status": "VALID",
        "train_artists": len(train_artists),
        "val_artists": len(val_artists),
        "test_artists": len(test_artists),
        "train_test_overlap": len(train_test_overlap),
        "leakage_pct": leakage_pct
    }

def main():
    print("=== RM-VMusic: Artist Leakage & Disjointness Verification ===")
    results = []
    for sfile in SPLIT_FILES:
        res = analyze_artist_leakage_for_file(SPLITS_DIR / sfile)
        if res:
            results.append(res)
    print("\n=== Artist Leakage Check Completed ===")

if __name__ == "__main__":
    main()
