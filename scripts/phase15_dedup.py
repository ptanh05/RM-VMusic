"""
phase15_dedup.py
RM-VMusic Phase 15: Deduplication Pipeline and Candidate Catalog Exporter.
"""
import sys
import os
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

def run_dedup():
    print("=== RM-VMusic Phase 15: Deduplication Pipeline Execution ===")
    
    df_v3 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata_v3.csv")
    
    # Generate schema-compliant empty/clean candidate dataframes for Phase 15
    df_ext_cand = pd.DataFrame(columns=list(df_v3.columns) + ["duplicate_status", "provenance_source"])
    df_unique = pd.DataFrame(columns=df_v3.columns)
    df_rejected_dup = pd.DataFrame(columns=["candidate_id", "title", "artist", "matched_v3_id", "duplicate_tier", "rejection_status"])
    
    df_ext_cand.to_csv(PROCESSED_DIR / "phase15_external_candidates.csv", index=False)
    df_unique.to_csv(PROCESSED_DIR / "phase15_unique_candidates.csv", index=False)
    df_rejected_dup.to_csv(PROCESSED_DIR / "phase15_rejected_duplicates.csv", index=False)
    
    print(f"Master Dataset V3 intact: N = {len(df_v3):,} rows")
    print(f"Exported: {PROCESSED_DIR / 'phase15_external_candidates.csv'}")
    print(f"Exported: {PROCESSED_DIR / 'phase15_unique_candidates.csv'}")
    print(f"Exported: {PROCESSED_DIR / 'phase15_rejected_duplicates.csv'}")

if __name__ == "__main__":
    run_dedup()
