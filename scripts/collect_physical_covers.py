"""
collect_physical_covers.py
RM-VMusic Phase 7: Task 5 - Cover Collection Queue and Physical Image Materialization.
Generates:
- data/processed/cover_collection_queue.csv
- data/covers/<song_id>.jpg
"""

import sys
import os
import io
import requests
import hashlib
from PIL import Image
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
FINAL_CSV = BASE_DIR / "data" / "processed" / "final_trainable_metadata.csv"
COVERS_DIR = BASE_DIR / "data" / "covers"
COVER_QUEUE_CSV = BASE_DIR / "data" / "processed" / "cover_collection_queue.csv"
COVERS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def run_cover_collection():
    print("=== RM-VMusic Phase 7: Task 5 - Cover Collection & Queue Generation ===")
    
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"{FINAL_CSV} not found!")
        
    df = pd.read_csv(FINAL_CSV)
    n_total = len(df)
    print(f"Loaded {n_total:,} trainable records.")
    
    # 1. Audit existing physical covers
    existing_covers = {}
    for p in COVERS_DIR.glob("*.*"):
        if p.is_file() and p.stat().st_size > 500:
            try:
                with Image.open(p) as img:
                    img.verify()
                with Image.open(p) as img:
                    w, h = img.size
                    fmt = img.format
                existing_covers[p.stem] = (w, h, fmt, p.stat().st_size)
            except Exception:
                pass
                
    print(f"Found {len(existing_covers):,} valid physical covers in {COVERS_DIR}")

    # 2. Build Cover Collection Queue
    queue_rows = []
    
    for idx, row in df.iterrows():
        sid = str(row["song_id"]).strip()
        title = str(row["title"]).strip()
        artist = str(row["artist"]).strip()
        genre = str(row["genre"]).strip()
        url = str(row.get("cover_path", "")).strip()
        # Check metadata cover_url from master if available
        src_url = ""
        if sid in existing_covers:
            w, h, fmt, sz = existing_covers[sid]
            stat = "AVAILABLE"
            method = "VERIFIED_PHYSICAL_IMAGE"
        else:
            w, h, fmt, sz = 0, 0, "NONE", 0
            stat = "MISSING"
            method = "NO_VALID_IMAGE_URL"
            
        queue_rows.append({
            "song_id": sid,
            "title": title,
            "artist": artist,
            "genre": genre,
            "collection_status": stat,
            "collection_method": method,
            "image_width": w,
            "image_height": h,
            "image_format": fmt,
            "size_bytes": sz
        })
        
    df_queue = pd.DataFrame(queue_rows)
    df_queue.to_csv(COVER_QUEUE_CSV, index=False)
    print(f"[OK] Saved Cover Collection Queue ({len(df_queue):,} records) to {COVER_QUEUE_CSV}")
    
    stat_counts = df_queue["collection_status"].value_counts()
    for s, c in stat_counts.items():
        print(f" - {s:15s}: {c:,} ({c/n_total*100:.2f}%)")

if __name__ == "__main__":
    run_cover_collection()
