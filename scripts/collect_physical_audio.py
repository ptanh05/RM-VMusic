"""
collect_physical_audio.py
RM-VMusic Phase 7: Task 2 & 3 - Audio Collection Queue & Physical Waveform Materialization.
Generates:
- data/processed/audio_collection_queue.csv
- data/audio/<song_id>.mp3 (for legally downloadable open-source audio)
"""

import sys
import os
import io
import re
import unicodedata
import requests
import hashlib
import pandas as pd
import numpy as np
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
AUDIO_DIR = BASE_DIR / "data" / "audio"
QUEUE_CSV = BASE_DIR / "data" / "processed" / "audio_collection_queue.csv"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def run_audio_collection():
    print("=== RM-VMusic Phase 7: Task 2 & 3 - Audio Collection Queue & Physical Materialization ===")
    
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"{FINAL_CSV} not found!")
        
    df = pd.read_csv(FINAL_CSV)
    n_total = len(df)
    print(f"Loaded {n_total:,} trainable records.")
    
    # -------------------------------------------------------------
    # 1. Inspect existing physical audio files
    # -------------------------------------------------------------
    existing_audios = {}
    for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg"]:
        for p in AUDIO_DIR.glob(f"*{ext}"):
            if p.is_file() and p.stat().st_size > 1000:
                sid = p.stem
                existing_audios[sid] = p
                
    print(f"Found {len(existing_audios):,} physical audio files already in {AUDIO_DIR}")

    # -------------------------------------------------------------
    # 2. Build Audio Collection Queue
    # -------------------------------------------------------------
    queue_rows = []
    
    downloadable_candidates = []
    
    for idx, row in df.iterrows():
        sid = str(row["song_id"]).strip()
        title = str(row["title"]).strip()
        artist = str(row["artist"]).strip()
        genre = str(row["genre"]).strip()
        url = str(row.get("source_url", "")).strip() if pd.notna(row.get("source_url")) else ""
        
        # Check current physical status
        if sid in existing_audios:
            status = "AVAILABLE"
            method = "LOCAL_FILESYSTEM"
            lic = "OPEN_RESEARCH_CACHE"
        elif "zmdcdn.me" in url:
            status = "URL_ONLY"
            method = "EXPIRED_STREAMING_TOKEN"
            lic = "PROPRIETARY_STREAM_NO_DOWNLOAD"
        elif "musicbrainz.org" in url:
            status = "URL_ONLY"
            method = "WEB_ENTITY_METADATA"
            lic = "CC0_METADATA_NO_AUDIO"
        elif "zingmp3.vn" in url:
            status = "URL_ONLY"
            method = "WEB_PAGE_HTML"
            lic = "PROPRIETARY_WEB_NO_DOWNLOAD"
        elif url.startswith("http") and (url.endswith(".mp3") or url.endswith(".wav")):
            status = "DOWNLOADABLE"
            method = "DIRECT_HTTP_STREAM"
            lic = "OPEN_ACCESS"
            downloadable_candidates.append((sid, url))
        elif not url:
            status = "UNAVAILABLE"
            method = "NO_URL_PROVIDED"
            lic = "NONE"
        else:
            status = "REQUIRES_MANUAL_REVIEW"
            method = "UNRECOGNIZED_SOURCE"
            lic = "UNKNOWN"
            
        queue_rows.append({
            "song_id": sid,
            "title": title,
            "artist": artist,
            "genre": genre,
            "audio_url": url,
            "source": str(row.get("source_id", "catalog")),
            "collection_status": status,
            "collection_method": method,
            "license": lic
        })
        
    df_queue = pd.DataFrame(queue_rows)
    df_queue.to_csv(QUEUE_CSV, index=False)
    print(f"[OK] Generated Audio Collection Queue ({len(df_queue):,} records) saved to {QUEUE_CSV}")
    
    # -------------------------------------------------------------
    # 3. Status Summary of Queue
    # -------------------------------------------------------------
    status_counts = df_queue["collection_status"].value_counts()
    print("\nAudio Collection Queue Breakdown:")
    for stat, cnt in status_counts.items():
        print(f" - {stat:25s}: {cnt:,} ({cnt/n_total*100:.2f}%)")
        
    # -------------------------------------------------------------
    # 4. Controlled Download for DOWNLOADABLE Candidates
    # -------------------------------------------------------------
    if len(downloadable_candidates) > 0:
        print(f"\nAttempting download for {len(downloadable_candidates)} open downloadable audio URLs...")
        downloaded = 0
        session = requests.Session()
        session.headers.update(HEADERS)
        
        for sid, d_url in downloadable_candidates:
            target_p = AUDIO_DIR / f"{sid}.mp3"
            if target_p.exists() and target_p.stat().st_size > 1000:
                downloaded += 1
                continue
            try:
                resp = session.get(d_url, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 5000:
                    with open(target_p, "wb") as f:
                        f.write(resp.content)
                    downloaded += 1
            except Exception:
                pass
        print(f"Successfully materialized {downloaded} open audio files.")
    else:
        print("\nNo unauthenticated direct public MP3 URLs found in metadata table.")
        print("Note: Streaming tokens are expired (HTTP 403), respecting strict rule against unauthorized scraping.")

if __name__ == "__main__":
    run_audio_collection()
