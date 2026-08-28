"""
audit_covers.py
RM-VMusic Phase 6: Systematic Audit of Physical Cover Art Images.
Generates:
- reports/cover_quality_report.csv
"""

import sys
import os
import io
import hashlib
from PIL import Image
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
TRAINABLE_CSV = BASE_DIR / "data" / "processed" / "trainable_metadata.csv"
COVERS_DIR = BASE_DIR / "data" / "covers"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GENRES = [
    "POP_BALLAD",
    "BOLERO_TRUTINH",
    "INSTRUMENTAL",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "DANCE_EDM",
    "REVOLUTIONARY",
    "NHAC_TRINH",
    "ROCK",
    "RB_SOUL",
    "CHILDREN"
]

def run_cover_audit():
    print("=== RM-VMusic Phase 6: Systematic Physical Cover Art Audit ===")
    df = pd.read_csv(TRAINABLE_CSV)
    n_total = len(df)
    
    cover_records = []
    seen_img_hashes = {}
    
    for idx, row in df.iterrows():
        song_id = str(row["song_id"]).strip()
        genre = str(row["genre"]).strip()
        
        file_found = False
        file_valid = False
        w, h = 0, 0
        size_b = 0
        is_dup_img = False
        
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = COVERS_DIR / f"{song_id}{ext}"
            if p.exists() and p.stat().st_size > 500:
                file_found = True
                size_b = p.stat().st_size
                try:
                    with Image.open(p) as img:
                        img.verify()
                    with Image.open(p) as img:
                        w, h = img.size
                    file_valid = True
                    
                    # Hash check
                    with open(p, "rb") as f:
                        im_bytes = f.read()
                    im_hash = hashlib.md5(im_bytes).hexdigest()
                    if im_hash in seen_img_hashes:
                        is_dup_img = True
                    else:
                        seen_img_hashes[im_hash] = song_id
                except Exception:
                    file_valid = False
                break
                
        cover_records.append({
            "song_id": song_id,
            "genre": genre,
            "physical_cover_exists": file_found,
            "physical_cover_valid": file_valid,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(1, h), 3) if file_valid else 0.0,
            "size_bytes": size_b,
            "is_duplicate_image": is_dup_img
        })
        
    df_cov = pd.DataFrame(cover_records)
    
    # Genre level aggregation
    genre_stats = []
    for g in GENRES:
        df_g = df_cov[df_cov["genre"] == g]
        n_g = len(df_g)
        n_phys = df_g["physical_cover_valid"].sum()
        avg_w = df_g[df_g["physical_cover_valid"]]["width"].mean() if n_phys > 0 else 0
        avg_h = df_g[df_g["physical_cover_valid"]]["height"].mean() if n_phys > 0 else 0
        avg_sz = df_g[df_g["physical_cover_valid"]]["size_bytes"].mean() / 1024 if n_phys > 0 else 0
        
        genre_stats.append({
            "Genre": g,
            "Total_Samples": n_g,
            "Physical_Cover_Count": int(n_phys),
            "Physical_Cover_Pct": round(n_phys / n_g * 100, 2),
            "Avg_Resolution": f"{int(avg_w)}x{int(avg_h)}" if n_phys > 0 else "N/A",
            "Avg_Size_KB": round(avg_sz, 1),
            "Physical_Gap": int(n_g - n_phys)
        })
        
    df_genre = pd.DataFrame(genre_stats)
    csv_path = REPORTS_DIR / "cover_quality_report.csv"
    df_genre.to_csv(csv_path, index=False)
    print(f"[OK] Saved {csv_path}")

if __name__ == "__main__":
    run_cover_audit()
