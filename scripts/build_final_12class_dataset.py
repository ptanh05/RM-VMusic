"""
build_final_12class_dataset.py
RM-VMusic Phase 7: Task 9, 10 & 11 - 12-Class Taxonomy Construction & Physical Modality Partitioning.
Output: data/processed/final_12class_metadata.csv
"""

import sys
import os
import io
import re
import unicodedata
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
FINAL_11_CSV = BASE_DIR / "data" / "processed" / "final_trainable_metadata.csv"
MASTER_CSV = BASE_DIR / "data" / "processed" / "master_metadata.csv"
OUTPUT_12_CSV = BASE_DIR / "data" / "processed" / "final_12class_metadata.csv"
AUDIO_DIR = BASE_DIR / "data" / "audio"
COVERS_DIR = BASE_DIR / "data" / "covers"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GENRES_12 = [
    "POP_BALLAD",
    "BOLERO_TRUTINH",
    "DANCE_EDM",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "CHILDREN",
    "REVOLUTIONARY",
    "RB_SOUL",
    "NHAC_TRINH",
    "INSTRUMENTAL",
    "ROCK",
    "OTHER"
]

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def run_build_12class_dataset():
    print("=== RM-VMusic Phase 7: Building Final 12-Class Physical Dataset ===")
    
    df_11 = pd.read_csv(FINAL_11_CSV)
    df_master = pd.read_csv(MASTER_CSV)
    print(f"Loaded 11-class trainable: {len(df_11):,} records")
    print(f"Loaded master catalog: {len(df_master):,} records")
    
    # -------------------------------------------------------------
    # 1. Identify and Annotate Positive OTHER Class Samples
    # -------------------------------------------------------------
    master_map = df_master.set_index("song_id").to_dict(orient="index")
    existing_sids = set(df_11["song_id"])
    
    other_candidates = []
    
    for sid, row in master_map.items():
        if sid in existing_sids:
            continue
        sg = str(row.get("source_genre", "")).lower().strip()
        g = str(row.get("genre", "")).strip()
        
        # Check for explicit out-of-taxonomy genre labels
        other_reason = ""
        if "tôn giáo" in sg or "đạo" in sg:
            other_reason = "Religious / Sacred spiritual Vietnamese music"
        elif "phim" in sg or "soundtrack" in sg or "ost" in sg:
            other_reason = "Film Soundtrack / OST composition"
        elif "country" in sg:
            other_reason = "Country style"
        elif "jazz" in sg:
            other_reason = "Jazz instrumental / vocal"
        elif "choral" in sg or "thính phòng" in sg:
            other_reason = "Choral / Classical chamber music"
            
        if other_reason:
            other_candidates.append({
                "song_id": sid,
                "title": str(row.get("title", "")),
                "artist": str(row.get("artist", "")),
                "artist_id": str(row.get("artist_id", f"art_{normalize_text(row.get('artist', ''))}")),
                "genre": "OTHER",
                "label_source": "curated_out_of_taxonomy",
                "label_confidence": 0.90,
                "other_reason": other_reason,
                "tier": "TIER_B",
                "release_year": row.get("release_year", None),
                "raw_genre": sg,
                "source_genre": sg,
                "source_url": str(row.get("audio_url", "")),
                "source_id": str(row.get("source_id", sid))
            })
            
    print(f"Found and annotated {len(other_candidates):,} positive verified OTHER class samples.")
    
    # -------------------------------------------------------------
    # 2. Assemble Master 12-Class Dataset
    # -------------------------------------------------------------
    final_12_rows = []
    
    # Add existing 11-class records
    for idx, row in df_11.iterrows():
        sid = str(row["song_id"]).strip()
        
        # Audio / Lyrics / Cover Physical Status
        has_a = (AUDIO_DIR / f"{sid}.mp3").exists() or (AUDIO_DIR / f"{sid}.wav").exists()
        has_l = (LYRICS_DIR / f"{sid}.txt").exists() and (LYRICS_DIR / f"{sid}.txt").stat().st_size > 10
        has_c = False
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (COVERS_DIR / f"{sid}{ext}").exists() and (COVERS_DIR / f"{sid}{ext}").stat().st_size > 500:
                has_c = True
                break
                
        # Modality State
        if has_a and has_l and has_c:
            mod_state = "FULL_MULTIMODAL"
        elif has_a and has_l:
            mod_state = "AUDIO_LYRICS"
        elif has_a and has_c:
            mod_state = "AUDIO_COVER"
        elif has_l and has_c:
            mod_state = "LYRICS_COVER"
        elif has_a:
            mod_state = "AUDIO_ONLY"
        elif has_l:
            mod_state = "LYRICS_ONLY"
        elif has_c:
            mod_state = "COVER_ONLY"
        else:
            mod_state = "NO_PHYSICAL_MODALITY"
            
        ry = row.get("release_year")
        year_stat = "VERIFIED" if pd.notna(ry) and str(ry).strip() not in ("", "nan", "None") else "UNVERIFIED"
        
        final_12_rows.append({
            "song_id": sid,
            "title": str(row["title"]),
            "artist": str(row["artist"]),
            "artist_id": str(row["artist_id"]),
            "genre": str(row["genre"]),
            "label_source": str(row.get("label_source", "verified_catalog")),
            "label_confidence": float(row.get("label_confidence", 1.0)),
            "other_reason": "N/A (Standard 11 Taxonomy)",
            "audio_path": f"data/audio/{sid}.mp3" if has_a else "",
            "audio_status": "AVAILABLE" if has_a else "MISSING",
            "lyrics_path": f"data/lyrics/{sid}.txt" if has_l else "",
            "lyrics_status": "AVAILABLE" if has_l else "MISSING",
            "cover_path": f"data/covers/{sid}.jpg" if has_c else "",
            "cover_status": "AVAILABLE" if has_c else "MISSING",
            "release_year": ry,
            "year_status": year_stat,
            "tier": str(row.get("tier", "TIER_A")),
            "modality_state": mod_state
        })
        
    # Add OTHER candidates
    for cand in other_candidates:
        sid = cand["song_id"]
        has_a = (AUDIO_DIR / f"{sid}.mp3").exists() or (AUDIO_DIR / f"{sid}.wav").exists()
        has_l = (LYRICS_DIR / f"{sid}.txt").exists() and (LYRICS_DIR / f"{sid}.txt").stat().st_size > 10
        has_c = False
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (COVERS_DIR / f"{sid}{ext}").exists() and (COVERS_DIR / f"{sid}{ext}").stat().st_size > 500:
                has_c = True
                break
                
        if has_a and has_l and has_c:
            mod_state = "FULL_MULTIMODAL"
        elif has_a and has_l:
            mod_state = "AUDIO_LYRICS"
        elif has_a and has_c:
            mod_state = "AUDIO_COVER"
        elif has_l and has_c:
            mod_state = "LYRICS_COVER"
        elif has_a:
            mod_state = "AUDIO_ONLY"
        elif has_l:
            mod_state = "LYRICS_ONLY"
        elif has_c:
            mod_state = "COVER_ONLY"
        else:
            mod_state = "NO_PHYSICAL_MODALITY"
            
        ry = cand.get("release_year")
        year_stat = "VERIFIED" if pd.notna(ry) and str(ry).strip() not in ("", "nan", "None") else "UNVERIFIED"
        
        final_12_rows.append({
            "song_id": sid,
            "title": cand["title"],
            "artist": cand["artist"],
            "artist_id": cand["artist_id"],
            "genre": "OTHER",
            "label_source": cand["label_source"],
            "label_confidence": cand["label_confidence"],
            "other_reason": cand["other_reason"],
            "audio_path": f"data/audio/{sid}.mp3" if has_a else "",
            "audio_status": "AVAILABLE" if has_a else "MISSING",
            "lyrics_path": f"data/lyrics/{sid}.txt" if has_l else "",
            "lyrics_status": "AVAILABLE" if has_l else "MISSING",
            "cover_path": f"data/covers/{sid}.jpg" if has_c else "",
            "cover_status": "AVAILABLE" if has_c else "MISSING",
            "release_year": ry,
            "year_status": year_stat,
            "tier": cand["tier"],
            "modality_state": mod_state
        })
        
    df_12 = pd.DataFrame(final_12_rows)
    df_12.to_csv(OUTPUT_12_CSV, index=False)
    print(f"\n[OK] Successfully saved {len(df_12):,} records (12 classes) to {OUTPUT_12_CSV}")
    
    # Modality State Breakdown
    print("\nPhysical Modality States Breakdown:")
    for st, cnt in df_12["modality_state"].value_counts().items():
        print(f" - {st:22s}: {cnt:,} ({cnt/len(df_12)*100:.2f}%)")
        
    # Genre Distribution
    print("\n12-Class Genre Distribution:")
    for g, cnt in df_12["genre"].value_counts().items():
        print(f" - {g:20s}: {cnt:,} ({cnt/len(df_12)*100:.2f}%)")

if __name__ == "__main__":
    run_build_12class_dataset()
