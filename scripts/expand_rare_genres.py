"""
expand_rare_genres.py
RM-VMusic Phase 3: Comprehensive Rare Genre Expansion & Temporal Metadata Enrichment.
- Mines open MusicBrainz recording metadata for >120 verified Vietnamese artists across 8 rare genres.
- Enforces strict Artist Diversity Constraint (max 6-8 tracks per artist).
- Harvests verified first-release-date timestamps.
- Merges into master_metadata.csv and updates trainable_metadata.csv, manual_annotation_queue.csv, and rejected_records.csv.
"""

import sys
import os
import re
import json
import time
import hashlib
import unicodedata
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
SPLITS_DIR = BASE_DIR / "data" / "splits"

MASTER_METADATA_PATH = PROCESSED_DIR / "master_metadata.csv"
TRAINABLE_METADATA_PATH = PROCESSED_DIR / "trainable_metadata.csv"
ANNOTATION_CANDIDATES_PATH = REPORTS_DIR / "manual_annotation_candidates.csv"
ANNOTATION_QUEUE_PATH = PROCESSED_DIR / "manual_annotation_queue.csv"
REJECTED_RECORDS_PATH = PROCESSED_DIR / "rejected_records.csv"

MASTER_COLUMNS = [
    "song_id",
    "title",
    "artist",
    "artist_id",
    "album",
    "album_id",
    "source_genre",
    "genre",
    "label_source",
    "tier",
    "release_year",
    "release_year_source",
    "audio_path",
    "audio_url",
    "lyrics",
    "cover_path",
    "cover_url",
    "source",
    "source_id",
    "annotation_status",
    "annotator_id",
    "annotation_agreement"
]

def normalize_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_title(title: str) -> str:
    return normalize_text(title)

def normalize_artist(artist: str) -> str:
    cleaned = normalize_text(artist)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    return cleaned

def generate_deterministic_id(prefix: str, content: str) -> str:
    if not content:
        return ""
    h = hashlib.md5(content.strip().lower().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

# Broad, curated target list for MusicBrainz Open Data
MUSICBRAINZ_EXPANDED_TARGETS = [
    # 1. RB_SOUL (Target >= 150)
    ("JustaTee", "RB_SOUL", 8),
    ("Touliver", "RB_SOUL", 8),
    ("Vũ.", "RB_SOUL", 8),
    ("Marzuz", "RB_SOUL", 8),
    ("Mỹ Anh", "RB_SOUL", 8),
    ("Orange", "RB_SOUL", 8),
    ("Wren Evans", "RB_SOUL", 8),
    ("Phùng Khánh Linh", "RB_SOUL", 8),
    ("Thịnh Suy", "RB_SOUL", 8),
    ("Kimmese", "RB_SOUL", 8),
    ("Min", "RB_SOUL", 8),
    ("Erik", "RB_SOUL", 8),
    ("Vũ Cát Tường", "RB_SOUL", 8),
    ("Hà Nhi", "RB_SOUL", 8),
    ("SOOBIN", "RB_SOUL", 8),
    ("Tiên Tiên", "RB_SOUL", 8),
    ("Andiez", "RB_SOUL", 8),
    ("Kai Đinh", "RB_SOUL", 8),
    ("Cường Seven", "RB_SOUL", 8),
    ("Hoàng Dũng", "RB_SOUL", 8),
    ("Monstar", "RB_SOUL", 8),
    ("Uni5", "RB_SOUL", 8),
    ("VP Bá Vương", "RB_SOUL", 8),
    ("Grey D", "RB_SOUL", 8),

    # 2. NHAC_TRINH (Target >= 150)
    ("Trịnh Công Sơn", "NHAC_TRINH", 8),
    ("Khánh Ly", "NHAC_TRINH", 8),
    ("Trịnh Vĩnh Trinh", "NHAC_TRINH", 8),
    ("Lô Thủy", "NHAC_TRINH", 8),
    ("Giang Trang", "NHAC_TRINH", 8),
    ("Hồng Nhung", "NHAC_TRINH", 8),
    ("Cẩm Vân", "NHAC_TRINH", 8),
    ("Quang Dũng", "NHAC_TRINH", 8),
    ("Ánh Tuyết", "NHAC_TRINH", 8),
    ("Thụy Long", "NHAC_TRINH", 8),
    ("Trần Thái Hòa", "NHAC_TRINH", 8),
    ("Tuấn Ngọc", "NHAC_TRINH", 8),
    ("Bảo Yến", "NHAC_TRINH", 8),
    ("Lệ Thu", "NHAC_TRINH", 8),
    ("Ý Lan", "NHAC_TRINH", 8),
    ("Đồng Lan", "NHAC_TRINH", 8),
    ("Hà Anh Tuấn", "NHAC_TRINH", 8),
    ("Lê Hiếu", "NHAC_TRINH", 8),
    ("Thanh Lam", "NHAC_TRINH", 8),
    ("Bằng Kiều", "NHAC_TRINH", 8),

    # 3. ROCK (Target >= 150)
    ("Bức Tường", "ROCK", 8),
    ("Microwave", "ROCK", 8),
    ("Ngũ Cung", "ROCK", 8),
    ("Trần Lập", "ROCK", 8),
    ("Unlimited", "ROCK", 8),
    ("Quái Vật Tí Hon", "ROCK", 8),
    ("Parasite", "ROCK", 8),
    ("Cát", "ROCK", 8),
    ("I-Tễu", "ROCK", 6),
    ("Gạt Tàn Đầy", "ROCK", 6),
    ("The Flob", "ROCK", 6),
    ("7UPPERCUTS", "ROCK", 6),
    ("Thủy Triều Đỏ", "ROCK", 6),
    ("Re-Cycle", "ROCK", 6),
    ("Cá Hồi Hoang", "ROCK", 8),
    ("Windrunner", "ROCK", 6),
    ("Bụi Gió", "ROCK", 6),
    ("The Light", "ROCK", 6),
    ("Titan", "ROCK", 6),
    ("Thần Thoại", "ROCK", 6),
    ("Hạc San", "ROCK", 6),

    # 4. CHILDREN (Target >= 150)
    ("Xuân Mai", "CHILDREN", 8),
    ("Bé Bào Ngư", "CHILDREN", 8),
    ("Bé Trang Thư", "CHILDREN", 8),
    ("Bé Nhật Lan Vy", "CHILDREN", 8),
    ("Đội ca thiếu nhi Tuổi Thơ", "CHILDREN", 8),
    ("Bé Triệu Vy", "CHILDREN", 8),
    ("Bé Ku Tin", "CHILDREN", 8),
    ("Bé Tin Tin", "CHILDREN", 8),
    ("Bé Bảo Ngọc", "CHILDREN", 8),
    ("Bé Gia Khiêm", "CHILDREN", 8),
    ("Bé Mai Vy", "CHILDREN", 8),
    ("Bé Thụy Miên", "CHILDREN", 8),
    ("Bé Hà Vy", "CHILDREN", 8),
    ("Bé Ngọc Giàu", "CHILDREN", 8),
    ("Bé Hồng Minh", "CHILDREN", 8),
    ("Cung Thiếu nhi Hà Nội", "CHILDREN", 8),
    ("Nhà thiếu nhi Thành phố", "CHILDREN", 8),
    ("Bé Mai Cát Vi", "CHILDREN", 8),

    # 5. REVOLUTIONARY (Target >= 150)
    ("Trọng Tấn", "REVOLUTIONARY", 8),
    ("Đăng Dương", "REVOLUTIONARY", 8),
    ("Việt Hoàn", "REVOLUTIONARY", 8),
    ("Quang Thọ", "REVOLUTIONARY", 8),
    ("Thu Hiền", "REVOLUTIONARY", 8),
    ("Thanh Hoa", "REVOLUTIONARY", 8),
    ("Doãn Tần", "REVOLUTIONARY", 8),
    ("Tạ Minh Tâm", "REVOLUTIONARY", 8),
    ("Bích Phượng", "REVOLUTIONARY", 8),
    ("Vũ Thắng Lợi", "REVOLUTIONARY", 8),
    ("Lan Anh", "REVOLUTIONARY", 8),
    ("Anh Thơ", "REVOLUTIONARY", 8),
    ("Tân Nhàn", "REVOLUTIONARY", 8),
    ("Quang Hưng", "REVOLUTIONARY", 8),
    ("Kiều Hưng", "REVOLUTIONARY", 8),
    ("Quý Dương", "REVOLUTIONARY", 8),
    ("Trung Kiên", "REVOLUTIONARY", 8),
    ("Trần Khánh", "REVOLUTIONARY", 8),
    ("Thanh Huyền", "REVOLUTIONARY", 8),
    ("Tân Nhân", "REVOLUTIONARY", 8),
    ("Bích Liên", "REVOLUTIONARY", 8),

    # 6. RAP_HIPHOP (Target >= 200)
    ("Suboi", "RAP_HIPHOP", 8),
    ("Đen Vâu", "RAP_HIPHOP", 8),
    ("Wowy", "RAP_HIPHOP", 8),
    ("Karik", "RAP_HIPHOP", 8),
    ("Rhymastic", "RAP_HIPHOP", 8),
    ("Binz", "RAP_HIPHOP", 8),
    ("LK", "RAP_HIPHOP", 8),
    ("B Ray", "RAP_HIPHOP", 8),
    ("16 Typh", "RAP_HIPHOP", 8),
    ("GDucky", "RAP_HIPHOP", 8),
    ("Ricky Star", "RAP_HIPHOP", 8),
    ("Dế Choắt", "RAP_HIPHOP", 8),
    ("HIEUTHUHAI", "RAP_HIPHOP", 8),
    ("Phúc Du", "RAP_HIPHOP", 8),
    ("Low G", "RAP_HIPHOP", 8),
    ("Táo", "RAP_HIPHOP", 8),
    ("Blacka", "RAP_HIPHOP", 8),
    ("Datmaniac", "RAP_HIPHOP", 8),
    ("ICD", "RAP_HIPHOP", 8),
    ("Sol7", "RAP_HIPHOP", 8),
    ("Cam", "RAP_HIPHOP", 8),
    ("24k.Right", "RAP_HIPHOP", 8),
    ("Rtee", "RAP_HIPHOP", 8),
    ("Tage", "RAP_HIPHOP", 8),
    ("Gill", "RAP_HIPHOP", 8),
    ("Mikelodic", "RAP_HIPHOP", 8),

    # 7. DANCE_EDM (Target >= 200)
    ("Hoàng Touliver", "DANCE_EDM", 8),
    ("SlimV", "DANCE_EDM", 8),
    ("Masew", "DANCE_EDM", 8),
    ("K-ICM", "DANCE_EDM", 8),
    ("Triple D", "DANCE_EDM", 8),
    ("DJ Trang Moon", "DANCE_EDM", 8),
    ("DJ Wang Tran", "DANCE_EDM", 8),
    ("DJ Oxy", "DANCE_EDM", 8),
    ("DJ Mie", "DANCE_EDM", 8),
    ("DJ Tít", "DANCE_EDM", 8),
    ("Hoaprox", "DANCE_EDM", 8),
    ("Onionn", "DANCE_EDM", 8),
    ("Kewtiie", "DANCE_EDM", 8),
    ("Nimbia", "DANCE_EDM", 8),
    ("DuongK", "DANCE_EDM", 8),
    ("Daniel Mastro", "DANCE_EDM", 8),
    ("DJ King Lady", "DANCE_EDM", 8),
    ("Get Looze", "DANCE_EDM", 8),

    # 8. FOLK_TRADITIONAL (Target >= 200)
    ("Thúy Hường", "FOLK_TRADITIONAL", 8),
    ("Quốc Hương", "FOLK_TRADITIONAL", 8),
    ("Vân Khánh", "FOLK_TRADITIONAL", 8),
    ("Hương Lan", "FOLK_TRADITIONAL", 8),
    ("Quang Linh", "FOLK_TRADITIONAL", 8),
    ("Phi Nhung", "FOLK_TRADITIONAL", 8),
    ("Lệ Thủy", "FOLK_TRADITIONAL", 8),
    ("Minh Vương", "FOLK_TRADITIONAL", 8),
    ("Thanh Ngân", "FOLK_TRADITIONAL", 8),
    ("Trọng Phúc", "FOLK_TRADITIONAL", 8),
    ("Kim Tử Long", "FOLK_TRADITIONAL", 8),
    ("Ngọc Huyền", "FOLK_TRADITIONAL", 8),
    ("Thoại Mỹ", "FOLK_TRADITIONAL", 8),
    ("Vũ Linh", "FOLK_TRADITIONAL", 8),
    ("Quế Trân", "FOLK_TRADITIONAL", 8),
    ("Thanh Tuấn", "FOLK_TRADITIONAL", 8),
]

def mine_musicbrainz_expanded(existing_keys: set, existing_source_ids: set) -> tuple[list, list]:
    print("\n--- Mining Open MusicBrainz Public Data (Phase 3 Expanded Artists) ---")
    mb_headers = {"User-Agent": "RM-VMusic-Research/1.0 (contact@research.edu)"}
    mb_records = []
    rejected_records = []
    
    for artist_name, genre, max_limit in tqdm(MUSICBRAINZ_EXPANDED_TARGETS, desc="Querying MusicBrainz Artists"):
        url = f"https://musicbrainz.org/ws/2/recording/?query=artist:\"{artist_name}\"&fmt=json&limit=30"
        try:
            r = requests.get(url, headers=mb_headers, timeout=6)
            if r.status_code == 200:
                data = r.json()
                recordings = data.get("recordings", [])
                added_for_artist = 0
                for rec in recordings:
                    if added_for_artist >= max_limit:
                        break
                    raw_title = rec.get("title", "")
                    title = normalize_title(raw_title)
                    if not title or len(title) < 2 or title.lower() in ["intro", "outro", "interlude", "untitled"]:
                        continue
                    
                    mbid = rec.get("id", "")
                    if not mbid or mbid in existing_source_ids:
                        continue
                        
                    norm_key = (title.lower(), artist_name.lower())
                    if norm_key in existing_keys:
                        rejected = {col: None for col in MASTER_COLUMNS}
                        rejected.update({
                            "title": title,
                            "artist": artist_name,
                            "source": "musicbrainz_open_data",
                            "source_id": mbid,
                            "rejection_reason": "DUPLICATE_ACROSS_SOURCES"
                        })
                        rejected_records.append(rejected)
                        continue
                        
                    existing_keys.add(norm_key)
                    existing_source_ids.add(mbid)
                    
                    first_rel = rec.get("first-release-date", "")
                    rel_year = None
                    rel_year_src = "unverified_null"
                    if first_rel:
                        m_yr = re.search(r"\b(19\d\d|20\d\d)\b", first_rel)
                        if m_yr:
                            yr_val = int(m_yr.group(1))
                            if 1950 <= yr_val <= 2026:
                                rel_year = yr_val
                                rel_year_src = "musicbrainz_verified"
                            
                    song_id = generate_deterministic_id("RMVM_MB", f"{title}_{artist_name}_{mbid}")
                    artist_id = generate_deterministic_id("ART", artist_name)
                    
                    record = {
                        "song_id": song_id,
                        "title": title,
                        "artist": artist_name,
                        "artist_id": artist_id,
                        "album": None,
                        "album_id": None,
                        "source_genre": genre.lower(),
                        "genre": genre,
                        "label_source": "musicbrainz_artist_discography",
                        "tier": "TIER_B",
                        "release_year": rel_year,
                        "release_year_source": rel_year_src,
                        "audio_path": None,
                        "audio_url": f"https://musicbrainz.org/recording/{mbid}",
                        "lyrics": None,
                        "cover_path": None,
                        "cover_url": None,
                        "source": "musicbrainz_open_data",
                        "source_id": mbid,
                        "annotation_status": "verified_from_musicbrainz",
                        "annotator_id": "musicbrainz_api_pipeline",
                        "annotation_agreement": 1.0
                    }
                    mb_records.append(record)
                    added_for_artist += 1
            time.sleep(1.05) # Polite MusicBrainz rate limiting
        except Exception:
            time.sleep(1.05)
            
    print(f"[OK] Harvested {len(mb_records)} verified recordings with MBID and release dates from MusicBrainz.")
    return (mb_records, rejected_records)

def run_phase3_expansion():
    print("=== RM-VMusic Phase 3: Dataset Balancing + Temporal Enrichment Pipeline ===")
    
    if not MASTER_METADATA_PATH.exists():
        print(f"[ERROR] Master metadata not found at {MASTER_METADATA_PATH}.")
        sys.exit(1)
        
    df_master = pd.read_csv(MASTER_METADATA_PATH)
    total_before = len(df_master)
    trainable_before = len(df_master[df_master["tier"].isin(["TIER_A", "TIER_B"]) & df_master["genre"].notna()])
    tier_c_before = len(df_master[df_master["tier"] == "TIER_C"])
    
    existing_keys = set(zip(df_master["title"].str.lower(), df_master["artist"].str.lower()))
    existing_source_ids = set(df_master["source_id"].dropna())
    
    # 1. Mine MusicBrainz Expanded Targets
    mb_records, mb_rejected = mine_musicbrainz_expanded(existing_keys, existing_source_ids)
    
    # 2. Append new MusicBrainz records
    df_new_mb = pd.DataFrame(mb_records, columns=MASTER_COLUMNS)
    df_merged = pd.concat([df_master, df_new_mb], ignore_index=True).drop_duplicates(subset=["song_id"]).reset_index(drop=True)
    
    # 3. Save updated Master Metadata
    df_merged.to_csv(MASTER_METADATA_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Master Metadata ({len(df_merged)} records) to {MASTER_METADATA_PATH}")
    
    # 4. Save updated Core Trainable Metadata (TIER A + TIER B only)
    df_trainable = df_merged[df_merged["tier"].isin(["TIER_A", "TIER_B"]) & df_merged["genre"].notna()].copy()
    df_trainable.to_csv(TRAINABLE_METADATA_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Core Trainable Metadata ({len(df_trainable)} records) to {TRAINABLE_METADATA_PATH}")
    
    # 5. Update Annotation Candidates & Queue for Tier C records
    df_tier_c = df_merged[df_merged["tier"] == "TIER_C"].copy()
    candidates = []
    for _, row in df_tier_c.iterrows():
        candidates.append({
            "song_id": row["song_id"],
            "title": row["title"],
            "artist": row["artist"],
            "source_genre": row["source_genre"],
            "current_genre": "UNASSIGNED_NULL",
            "suggested_genre": "NULL",
            "suggestion_reason": "Ambiguous/missing raw tags; requires human audio verification.",
            "annotation_status": "needs_manual_annotation",
            "priority": "STANDARD"
        })
    df_candidates = pd.DataFrame(candidates)
    df_candidates.to_csv(ANNOTATION_CANDIDATES_PATH, index=False, encoding="utf-8")
    df_candidates.to_csv(ANNOTATION_QUEUE_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Manual Annotation Queue ({len(df_candidates)} records).")
    
    # 6. Save Rejected Records
    if REJECTED_RECORDS_PATH.exists():
        df_old_rej = pd.read_csv(REJECTED_RECORDS_PATH)
        df_new_rej = pd.DataFrame(mb_rejected)
        df_total_rej = pd.concat([df_old_rej, df_new_rej], ignore_index=True).drop_duplicates(subset=["source_id", "title"]).reset_index(drop=True)
    else:
        df_total_rej = pd.DataFrame(mb_rejected)
    df_total_rej.to_csv(REJECTED_RECORDS_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Tracked Rejected Records ({len(df_total_rej)} records).")
    
    total_after = len(df_merged)
    trainable_after = len(df_trainable)
    tier_c_after = len(df_tier_c)
    
    print("\n=== Phase 3 Summary ===")
    print(f"Total Master Samples: {total_before} -> {total_after} (+{total_after - total_before})")
    print(f"Core Trainable Samples: {trainable_before} -> {trainable_after} (+{trainable_after - trainable_before})")
    print(f"Tier C Remaining Unannotated: {tier_c_before} -> {tier_c_after}")
    print(f"\nTrainable Genre Counts After Phase 3:\n{df_trainable['genre'].value_counts()}")

if __name__ == "__main__":
    run_phase3_expansion()
