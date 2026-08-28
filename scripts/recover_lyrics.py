"""
recover_lyrics.py
RM-VMusic Phase 6: Materialize embedded lyrics metadata into standardized physical text files.
Output path: data/lyrics/{song_id}.txt
"""

import sys
import os
import re
import unicodedata
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
TRAINABLE_CSV = BASE_DIR / "data" / "processed" / "trainable_metadata.csv"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
LYRICS_DIR.mkdir(parents=True, exist_ok=True)

def clean_lyrics_text(text):
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).strip())
    # Standardize line breaks
    text = re.sub(r"\r\n|\r", "\n", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def run_lyrics_materialization():
    print("=== RM-VMusic Phase 6: Materializing Physical Lyrics Files ===")
    
    if not TRAINABLE_CSV.exists():
        raise FileNotFoundError(f"{TRAINABLE_CSV} not found!")
        
    df = pd.read_csv(TRAINABLE_CSV)
    print(f"Loaded {len(df):,} trainable records from {TRAINABLE_CSV}")
    
    materialized_count = 0
    empty_count = 0
    total_words = 0
    total_chars = 0
    
    for idx, row in df.iterrows():
        song_id = str(row["song_id"]).strip()
        raw_lyrics = row.get("lyrics", "")
        cleaned = clean_lyrics_text(raw_lyrics)
        
        target_path = LYRICS_DIR / f"{song_id}.txt"
        
        if cleaned and len(cleaned) >= 10:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            materialized_count += 1
            words = len(cleaned.split())
            chars = len(cleaned)
            total_words += words
            total_chars += chars
        else:
            empty_count += 1
            # If an old file exists but current record is empty, remove it to prevent stale state
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
                    
    print(f"\n[OK] Materialization Complete!")
    print(f" - Physical files written to {LYRICS_DIR}: {materialized_count:,} / {len(df):,} ({materialized_count/len(df)*100:.2f}%)")
    print(f" - Empty / Missing lyrics records: {empty_count:,} ({empty_count/len(df)*100:.2f}%)")
    print(f" - Total Words Materialized: {total_words:,} (Average {total_words/max(1, materialized_count):.1f} words/song)")
    print(f" - Total Characters Materialized: {total_chars:,} (Average {total_chars/max(1, materialized_count):.1f} chars/song)")

if __name__ == "__main__":
    run_lyrics_materialization()
