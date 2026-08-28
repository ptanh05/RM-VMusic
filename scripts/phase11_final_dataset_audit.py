"""
phase11_final_dataset_audit.py
RM-VMusic Phase 11: Final Dataset V2 Quality Gate Auditor.
"""
import sys
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
SPLITS_V2_DIR = DATA_DIR = BASE_DIR / "data" / "splits" / "v2"
REPORTS_DIR = BASE_DIR / "reports"

def run_v2_quality_gate():
    print("=== RM-VMusic Phase 11: Dataset V2 Final Quality Gate ===")
    
    df_v1 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata.csv")
    df_v2 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata_v2.csv")
    
    # 1. Check preservation of V1
    v1_ids = set(df_v1["song_id"])
    v2_ids = set(df_v2["song_id"])
    assert v1_ids.issubset(v2_ids), "FAIL: Not all V1 IDs preserved in V2!"
    print(f"1. V1 Preservation: PASS (All {len(v1_ids):,} V1 tracks preserved in V2)")
    
    # 2. Check duplicate song_id
    dup_sid = df_v2["song_id"].duplicated().sum()
    assert dup_sid == 0, f"FAIL: {dup_sid} duplicate song_ids found in V2!"
    print(f"2. Duplicate Song ID: PASS (0 duplicates)")
    
    # 3. Check duplicate (title, artist)
    dup_ta = df_v2.duplicated(subset=["title", "artist"]).sum()
    assert dup_ta == 0, f"FAIL: {dup_ta} duplicate (title, artist) found in V2!"
    print(f"3. Duplicate (Title, Artist): PASS (0 duplicates)")
    
    # 4. Check Artist Disjoint Leakage
    tr_ad = pd.read_csv(SPLITS_V2_DIR / "v2_artist_train.csv")
    va_ad = pd.read_csv(SPLITS_V2_DIR / "v2_artist_val.csv")
    te_ad = pd.read_csv(SPLITS_V2_DIR / "v2_artist_test.csv")
    
    tr_art = set(tr_ad["artist"])
    va_art = set(va_ad["artist"])
    te_art = set(te_ad["artist"])
    
    l_tr_va = tr_art & va_art
    l_tr_te = tr_art & te_art
    l_va_te = va_art & te_art
    
    assert len(l_tr_va) == 0, f"FAIL: {len(l_tr_va)} artist leakage between Tr and Va!"
    assert len(l_tr_te) == 0, f"FAIL: {len(l_tr_te)} artist leakage between Tr and Te!"
    assert len(l_va_te) == 0, f"FAIL: {len(l_va_te)} artist leakage between Va and Te!"
    print(f"4. Artist Disjoint Leakage: PASS (Strict 0% Leakage: Tr∩Va=0, Tr∩Te=0, Va∩Te=0)")
    
    # 5. Check All 12 Classes Exist
    n_classes = df_v2["genre"].nunique()
    assert n_classes == 12, f"FAIL: Found {n_classes} classes instead of 12!"
    print(f"5. 12-Class Taxonomy: PASS (Exactly 12 verified classes present)")
    
    # 6. Check Provenance & Metadata Schema
    v2_only = df_v2[df_v2["dataset_version"] == "v2"]
    assert len(v2_only) == 54, f"FAIL: Found {len(v2_only)} v2 tracks instead of 54!"
    assert (v2_only["label_source"] == "vietlyrics_curated").all(), "FAIL: Incomplete label source in V2!"
    print(f"6. Provenance & Metadata Traceability: PASS (54 new tracks with 100% verified source)")
    
    print("\n========================================================")
    print("DATASET V2 QUALITY GATE: >>> ALL 6 CHECKS PASSED (PASS) <<<")
    print("========================================================")

if __name__ == "__main__":
    run_v2_quality_gate()
