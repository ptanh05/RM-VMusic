"""
check_duplicates.py
Analyzes and reports duplicates in RM-VMusic master and trainable metadata.
"""

import sys
import hashlib
import unicodedata
import re
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
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MASTER_METADATA_PATH = PROCESSED_DIR / "master_metadata.csv"
TRAINABLE_METADATA_PATH = PROCESSED_DIR / "trainable_metadata.csv"

def normalize_key_str(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    text = unicodedata.normalize("NFC", str(s))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def check_duplicates_on_df(df: pd.DataFrame, label: str):
    print(f"\n=== RM-VMusic: Deduplication Analysis ({label}) ===")
    total_records = len(df)
    print(f"Total Records Analyzed: {total_records}")
    
    dup_song_id = df[df.duplicated(subset=["song_id"], keep=False)]
    print(f"1. Duplicate Song IDs: {len(dup_song_id)} records")
    
    df["norm_title"] = df["title"].apply(normalize_key_str)
    df["norm_artist"] = df["artist"].apply(normalize_key_str)
    df["title_artist_pair"] = df["norm_title"] + "___" + df["norm_artist"]
    
    dup_title_artist = df[df.duplicated(subset=["title_artist_pair"], keep=False)]
    print(f"2. Duplicate (Title + Artist) pairs: {len(dup_title_artist)} records")
    
    valid_source_ids = df[df["source_id"].notna() & (df["source_id"] != "")]
    dup_source_id = valid_source_ids[valid_source_ids.duplicated(subset=["source_id"], keep=False)]
    print(f"3. Duplicate Source IDs (e.g. Zing ID): {len(dup_source_id)} records")
    
    valid_audio_urls = df[df["audio_url"].notna() & (df["audio_url"] != "")]
    dup_audio_url = valid_audio_urls[valid_audio_urls.duplicated(subset=["audio_url"], keep=False)]
    print(f"4. Duplicate Audio URLs: {len(dup_audio_url)} records")
    
    dup_rate = (len(dup_song_id) + len(dup_title_artist) + len(dup_source_id)) / (total_records * 3) * 100
    print(f"Estimated Duplicate Rate: {dup_rate:.2f}%")

def main():
    if MASTER_METADATA_PATH.exists():
        df_m = pd.read_csv(MASTER_METADATA_PATH)
        check_duplicates_on_df(df_m, "Master Metadata")
    if TRAINABLE_METADATA_PATH.exists():
        df_t = pd.read_csv(TRAINABLE_METADATA_PATH)
        check_duplicates_on_df(df_t, "Core Trainable Metadata")

if __name__ == "__main__":
    main()
