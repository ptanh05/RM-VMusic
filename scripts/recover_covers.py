"""
recover_covers.py
RM-VMusic Phase 6: Recover and validate physical cover art images from metadata URLs.
Output path: data/covers/{song_id}.jpg
Blocked/Failed: data/processed/recovery_blocked.csv
"""

import sys
import os
import time
import requests
import hashlib
from PIL import Image
import io
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
COVERS_DIR = BASE_DIR / "data" / "covers"
BLOCKED_CSV = BASE_DIR / "data" / "processed" / "recovery_blocked.csv"
COVERS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def run_cover_recovery():
    print("=== RM-VMusic Phase 6: Recovering Physical Cover Art Files ===")
    
    if not TRAINABLE_CSV.exists():
        raise FileNotFoundError(f"{TRAINABLE_CSV} not found!")
        
    df = pd.read_csv(TRAINABLE_CSV)
    print(f"Loaded {len(df):,} trainable records from {TRAINABLE_CSV}")
    
    df_with_covers = df[df["cover_url"].notna() & (df["cover_url"].str.strip() != "")].copy()
    print(f"Found {len(df_with_covers):,} records with cover_url in metadata.")
    
    recovered_count = 0
    failed_count = 0
    blocked_records = []
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for idx, (_, row) in enumerate(df_with_covers.iterrows()):
        song_id = str(row["song_id"]).strip()
        url = str(row["cover_url"]).strip()
        target_path = COVERS_DIR / f"{song_id}.jpg"
        
        # If valid physical file already exists, check validity
        if target_path.exists() and target_path.stat().st_size > 500:
            try:
                with Image.open(target_path) as img:
                    img.verify()
                recovered_count += 1
                continue
            except Exception:
                pass
                
        try:
            resp = session.get(url, timeout=6)
            if resp.status_code == 200 and len(resp.content) > 500:
                # Validate image header with PIL
                img = Image.open(io.BytesIO(resp.content))
                img.verify()
                
                # Re-open and save cleanly as JPEG
                img = Image.open(io.BytesIO(resp.content))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(target_path, "JPEG", quality=90)
                
                recovered_count += 1
            else:
                failed_count += 1
                blocked_records.append({
                    "song_id": song_id,
                    "modality": "cover",
                    "url": url,
                    "status_code": resp.status_code,
                    "reason": f"HTTP_{resp.status_code}_OR_EMPTY_BODY"
                })
        except Exception as e:
            failed_count += 1
            blocked_records.append({
                "song_id": song_id,
                "modality": "cover",
                "url": url,
                "status_code": 0,
                "reason": f"NETWORK_ERROR_{type(e).__name__}"
            })
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_with_covers):
            print(f" -> Progress: {idx+1}/{len(df_with_covers)} | Recovered: {recovered_count} | Blocked/Failed: {failed_count}")
            
    print(f"\n[OK] Cover Recovery Complete!")
    print(f" - Physical covers saved in {COVERS_DIR}: {recovered_count:,} / {len(df):,} ({recovered_count/len(df)*100:.2f}%)")
    print(f" - Failed/Blocked covers: {failed_count:,}")
    
    # Also log audio expired token URLs into recovery_blocked.csv
    df_audio = df[df["audio_url"].notna() & (df["audio_url"].str.strip() != "")]
    for _, row in df_audio.iterrows():
        song_id = str(row["song_id"]).strip()
        a_url = str(row["audio_url"]).strip()
        if "zmdcdn.me" in a_url:
            blocked_records.append({
                "song_id": song_id,
                "modality": "audio",
                "url": a_url,
                "status_code": 403,
                "reason": "EXPIRED_STREAMING_TOKEN_CDN_HTTP_403"
            })
        elif "musicbrainz.org" in a_url:
            blocked_records.append({
                "song_id": song_id,
                "modality": "audio",
                "url": a_url,
                "status_code": 200,
                "reason": "METADATA_RECORDING_ENTITY_NO_RAW_AUDIO_STREAM"
            })
        elif "zingmp3.vn" in a_url:
            blocked_records.append({
                "song_id": song_id,
                "modality": "audio",
                "url": a_url,
                "status_code": 200,
                "reason": "HTML_PAGE_NO_DIRECT_AUDIO_STREAM"
            })
            
    df_blocked = pd.DataFrame(blocked_records)
    df_blocked.to_csv(BLOCKED_CSV, index=False)
    print(f"[OK] Saved {len(df_blocked):,} blocked asset records with explicit provenance to {BLOCKED_CSV}")

if __name__ == "__main__":
    run_cover_recovery()
