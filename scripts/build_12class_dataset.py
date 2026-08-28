"""
build_12class_dataset.py
RM-VMusic Phase 7B: 12-Class Taxonomy Dataset Construction with Physical Asset Linking.

Produces:
- data/processed/final_12class_metadata.csv
- reports/final_12class_class_balance.csv
"""

import sys
import os
import io
import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

# UTF-8 output
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
    "INSTRUMENTAL",
    "RAP_HIPHOP",
    "FOLK_TRADITIONAL",
    "DANCE_EDM",
    "REVOLUTIONARY",
    "NHAC_TRINH",
    "ROCK",
    "RB_SOUL",
    "OTHER",
    "CHILDREN"
]

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def run_build_12class_dataset():
    print("=== RM-VMusic Phase 7B: Building Final 12-Class Physical Dataset ===")
    
    df_11 = pd.read_csv(FINAL_11_CSV)
    df_master = pd.read_csv(MASTER_CSV)
    print(f"Loaded 11-class trainable: {len(df_11):,} records")
    print(f"Loaded master catalog: {len(df_master):,} records")
    
    # 1. Identify and Annotate Positive OTHER Class Samples
    master_map = df_master.set_index("song_id").to_dict(orient="index")
    existing_sids = set(df_11["song_id"])
    
    other_candidates = []
    for sid, row in master_map.items():
        if sid in existing_sids:
            continue
        sg = str(row.get("source_genre", "")).lower().strip()
        
        other_reason = ""
        if "tôn giáo" in sg or "đạo" in sg:
            other_reason = "Religious / Sacred spiritual Vietnamese music"
        elif "phim" in sg or "soundtrack" in sg or "ost" in sg:
            other_reason = "Film Soundtrack / OST composition"
        elif "country" in sg:
            other_reason = "Country style"
        elif "tết" in sg:
            other_reason = "Holiday / Festival seasonal music"
        elif "jazz" in sg:
            other_reason = "Jazz instrumental / vocal"
            
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
                "source_genre": sg,
                "source_url": str(row.get("audio_url", "")),
                "source_id": str(row.get("source_id", sid))
            })
            
    print(f"Found and annotated {len(other_candidates):,} positive verified OTHER class samples.")
    
    # 2. Assemble Master 12-Class Dataset with Physical Paths
    final_12_rows = []
    
    # Add existing 11-class records
    for idx, row in df_11.iterrows():
        sid = str(row["song_id"]).strip()
        
        # Audio / Lyrics / Cover Physical Status
        has_a = (AUDIO_DIR / f"{sid}.mp3").exists() or (AUDIO_DIR / f"{sid}.wav").exists()
        has_l = (LYRICS_DIR / f"{sid}.txt").exists() and (LYRICS_DIR / f"{sid}.txt").stat().st_size > 10
        has_c = False
        cover_path_str = ""
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (COVERS_DIR / f"{sid}{ext}").exists() and (COVERS_DIR / f"{sid}{ext}").stat().st_size > 500:
                has_c = True
                cover_path_str = f"data/covers/{sid}{ext}"
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
            mod_state = "METADATA_ONLY"
            
        final_12_rows.append({
            "song_id": sid,
            "title": row["title"],
            "artist": row["artist"],
            "artist_id": row["artist_id"],
            "genre": row["genre"],
            "label_source": row.get("label_source", "curated_exact"),
            "label_confidence": float(row.get("label_confidence", 1.0)),
            "other_reason": "",
            "audio_path": f"data/audio/{sid}.mp3" if has_a else "",
            "audio_status": "verified_local" if has_a else "unavailable",
            "lyrics_path": f"data/lyrics/{sid}.txt" if has_l else "",
            "lyrics_status": "verified_local" if has_l else "unavailable",
            "cover_path": cover_path_str if has_c else "",
            "cover_status": "verified_local" if has_c else "unavailable",
            "release_year": row.get("release_year") if pd.notna(row.get("release_year")) else "",
            "year_status": "verified" if pd.notna(row.get("release_year")) and str(row.get("release_year")).strip() != "" else "missing",
            "tier": row.get("tier", "TIER_A"),
            "modality_state": mod_state
        })
        
    # Add OTHER candidates
    for cand in other_candidates:
        sid = cand["song_id"]
        has_a = (AUDIO_DIR / f"{sid}.mp3").exists() or (AUDIO_DIR / f"{sid}.wav").exists()
        has_l = (LYRICS_DIR / f"{sid}.txt").exists() and (LYRICS_DIR / f"{sid}.txt").stat().st_size > 10
        has_c = False
        cover_path_str = ""
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (COVERS_DIR / f"{sid}{ext}").exists() and (COVERS_DIR / f"{sid}{ext}").stat().st_size > 500:
                has_c = True
                cover_path_str = f"data/covers/{sid}{ext}"
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
            mod_state = "METADATA_ONLY"
            
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
            "audio_status": "verified_local" if has_a else "unavailable",
            "lyrics_path": f"data/lyrics/{sid}.txt" if has_l else "",
            "lyrics_status": "verified_local" if has_l else "unavailable",
            "cover_path": cover_path_str if has_c else "",
            "cover_status": "verified_local" if has_c else "unavailable",
            "release_year": cand.get("release_year") if pd.notna(cand.get("release_year")) else "",
            "year_status": "verified" if pd.notna(cand.get("release_year")) and str(cand.get("release_year")).strip() != "" else "missing",
            "tier": cand["tier"],
            "modality_state": mod_state
        })
        
    df_12 = pd.DataFrame(final_12_rows)
    df_12.to_csv(OUTPUT_12_CSV, index=False, encoding="utf-8")
    print(f"\nFinal 12-Class Dataset written to: {OUTPUT_12_CSV} ({len(df_12):,} tracks)")
    
    # 3. Class Balance Breakdown
    print("\n--- 12-Class Distribution Breakdown ---")
    cls_summary = []
    for g in GENRES_12:
        sub = df_12[df_12["genre"] == g]
        cnt = len(sub)
        pct = (cnt / len(df_12)) * 100.0
        art_cnt = sub["artist"].nunique()
        lyrics_cnt = (sub["lyrics_status"] == "verified_local").sum()
        cover_cnt = (sub["cover_status"] == "verified_local").sum()
        audio_cnt = (sub["audio_status"] == "verified_local").sum()
        cls_summary.append({
            "genre": g,
            "count": cnt,
            "pct": round(pct, 2),
            "unique_artists": art_cnt,
            "physical_lyrics": lyrics_cnt,
            "physical_covers": cover_cnt,
            "physical_audio": audio_cnt
        })
        print(f"  {g:<18}: {cnt:>5} ({pct:>5.2f}%) | Artists: {art_cnt:>4} | L: {lyrics_cnt:>4} | C: {cover_cnt:>3} | A: {audio_cnt}")
        
    df_cls = pd.DataFrame(cls_summary)
    df_cls.to_csv(REPORTS_DIR / "final_12class_class_balance.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    run_build_12class_dataset()
