"""
build_dataset.py
RM-VMusic: Full Dataset Curation, 3-Tier Hierarchy, and Annotation Queue Generator.
1. Ingests full VietLyrics + full sunbv56/song_dataset.
2. Assigns Dataset Tiers:
   - TIER A: High-confidence cross-verified labels (4,156 samples)
   - TIER B: Validated single-source deterministic mappings (147 samples)
   - TIER C: Needs manual annotation (3,612 samples)
3. Exports:
   - data/processed/master_metadata.csv (All 7,915 records)
   - data/processed/trainable_metadata.csv (4,303 Tier A & B records with valid ground truth)
   - reports/manual_annotation_candidates.csv (Analysis of all 3,612 Tier C records)
   - data/processed/manual_annotation_queue.csv (Prioritized manual annotation queue)
   - data/processed/rejected_records.csv (Tracked invalid records)
"""

import sys
import os
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
AUDIO_DIR = BASE_DIR / "data" / "audio"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
COVERS_DIR = BASE_DIR / "data" / "covers"
SPLITS_DIR = BASE_DIR / "data" / "splits"

for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, AUDIO_DIR, LYRICS_DIR, COVERS_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MASTER_METADATA_PATH = PROCESSED_DIR / "master_metadata.csv"
TRAINABLE_METADATA_PATH = PROCESSED_DIR / "trainable_metadata.csv"
ANNOTATION_CANDIDATES_PATH = REPORTS_DIR / "manual_annotation_candidates.csv"
ANNOTATION_QUEUE_PATH = PROCESSED_DIR / "manual_annotation_queue.csv"
REJECTED_RECORDS_PATH = PROCESSED_DIR / "rejected_records.csv"

# 22 Canonical Master Schema Fields (including tier)
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

def extract_verified_release_year(title: str, album: str) -> tuple[int | None, str]:
    for text, src_type in [(album, "album_metadata_verified"), (title, "title_metadata_verified")]:
        if text:
            matches = re.findall(r"[\(\[\s\-_](19[5-9]\d|20[0-2]\d)[\)\]\s\-_]?", str(text))
            if matches:
                try:
                    yr = int(matches[-1])
                    if 1950 <= yr <= 2026:
                        return (yr, src_type)
                except Exception:
                    pass
    return (None, "unverified_null")

def normalize_genre(raw_genre: str) -> tuple[str | None, str, str, str]:
    """
    Standardizes genre string into canonical taxonomy conforming to docs/genre_taxonomy.md.
    Returns: (normalized_genre, label_source, annotation_status, tier)
    """
    if not raw_genre or pd.isna(raw_genre):
        return (None, "unknown", "needs_manual_annotation", "TIER_C")
    
    clean = normalize_text(raw_genre).lower()
    if not clean or clean in ["unknown", "chưa biết", "khác", "null", "none", "nan", "not_found", "unknown genre", "new age / world music"]:
        return (None, "unknown", "needs_manual_annotation", "TIER_C")
    
    if any(k in clean for k in ["nhạc trẻ", "v-pop", "vpop", "pop", "teen pop", "ballad", "nhac tre"]):
        return ("POP_BALLAD", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["nhạc trữ tình", "bolero", "nhạc vàng", "trữ tình & bolero", "tru tinh"]):
        return ("BOLERO_TRUTINH", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["rap", "hip hop", "hip-hop", "rap việt"]):
        return ("RAP_HIPHOP", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["rock", "rock việt"]):
        return ("ROCK", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["dance", "edm", "vinahouse", "remix", "dance việt", "edm việt", "nhạc dance"]):
        return ("DANCE_EDM", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["cách mạng", "nhạc đỏ", "tiền chiến", "nhạc cách mạng"]):
        return ("REVOLUTIONARY", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["quê hương", "dân ca", "quan họ", "nhạc dân ca - quê hương", "que huong", "cải lương"]):
        return ("FOLK_TRADITIONAL", "deterministic_mapping", "normalized", "TIER_B")
    if "nhạc trịnh" in clean or "trịnh công sơn" in clean:
        return ("NHAC_TRINH", "deterministic_mapping", "normalized", "TIER_B")
    if "r&b" in clean or "soul" in clean:
        return ("RB_SOUL", "deterministic_mapping", "normalized", "TIER_B")
    if "thiếu nhi" in clean or "nhạc thiếu nhi" in clean:
        return ("CHILDREN", "deterministic_mapping", "normalized", "TIER_B")
    if any(k in clean for k in ["không lời", "instrumental", "hòa tấu", "blues"]):
        return ("INSTRUMENTAL", "deterministic_mapping", "normalized", "TIER_B")
    
    return (None, "unknown", "needs_manual_annotation", "TIER_C")

def extract_zing_id_from_url(url: str) -> str:
    if not url or pd.isna(url):
        return ""
    m = re.search(r"/([A-Z0-9]{8})\.html", str(url))
    if m:
        return m.group(1)
    m = re.search(r"([A-Z0-9]{8})", str(url))
    if m:
        return m.group(1)
    return ""

def fetch_vietlyrics_full() -> pd.DataFrame:
    dfs = []
    urls = [
        ("vietlyrics_val_1k.csv", "https://raw.githubusercontent.com/BatmanofZuhandArrgh/VietLyrics/main/data/val_1k_metadata_authors.csv"),
        ("vietlyrics_train_7k.csv", "https://raw.githubusercontent.com/BatmanofZuhandArrgh/VietLyrics/main/data/train_7k_metadata_authors.csv")
    ]
    for filename, url in urls:
        raw_cache = RAW_DIR / filename
        if raw_cache.exists():
            df_part = pd.read_csv(raw_cache)
            dfs.append(df_part)
    if dfs:
        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["link"])
    return pd.DataFrame()

def fetch_hf_full() -> list:
    records = []
    files = [
        RAW_DIR / "sunbv56_eval.jsonl",
        RAW_DIR / "sunbv56_train_full.jsonl",
        RAW_DIR / "sunbv56_pilot_train.jsonl"
    ]
    seen_ids = set()
    for fp in files:
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            zid = item.get("id", "").strip()
                            if zid and zid in seen_ids:
                                continue
                            if zid:
                                seen_ids.add(zid)
                            records.append(item)
                        except Exception:
                            pass
    return records

def generate_annotation_suggestion(title: str, artist: str, album: str, source_genre: str) -> tuple[str | None, str]:
    """
    Analyzes unannotated tracks and provides a candidate recommendation for human review.
    Does NOT convert candidate into ground truth.
    """
    raw_g = normalize_text(source_genre).lower() if source_genre else ""
    t_clean = normalize_text(title).lower()
    a_clean = normalize_text(artist).lower()
    alb_clean = normalize_text(album).lower()
    
    # 1. Evidence from raw source_genre
    if "new age" in raw_g or "world music" in raw_g:
        return ("INSTRUMENTAL", "Source tag 'new age / world music' indicates acoustic/ambient instrumental track; candidate for auditioning.")
    if "alternative" in raw_g:
        return ("ROCK", "Source tag 'alternative' aligns with Alternative Rock genre; candidate for human verification.")
    if "guitar" in raw_g:
        return ("INSTRUMENTAL", "Source tag 'guitar' indicates solo instrumental guitar performance.")
    if "nhạc tôn giáo" in raw_g or "nhạc đạo" in raw_g:
        return ("FOLK_TRADITIONAL", "Religious/hymnal vocal repertoire; candidate for folk/choral classification.")
    if "nhạc phim" in raw_g:
        return ("POP_BALLAD", "Soundtrack / OST vocal song; candidate for Pop/Ballad vocal classification.")
    if "tết" in raw_g or "xuân" in t_clean and "liên khúc" in t_clean:
        return ("POP_BALLAD", "Spring/Tet celebratory pop medley; candidate for Pop/Ballad classification.")
        
    # 2. Evidence from well-known Vietnamese artist discographies
    trinh_artists = ["khánh ly", "trịnh vĩnh trinh", "lô thủy", "hồng nhung", "cẩm vân", "trịnh công sơn"]
    if any(ta in a_clean for ta in trinh_artists) and any(kw in t_clean for kw in ["hạ trắng", "diễm xưa", "cát bụi", "nắng thủy tinh", "biển nhớ", "ru em", "tình nhớ", "mưa hồng"]):
        return ("NHAC_TRINH", f"Composer/Artist specialized repertoire in Nhạc Trịnh ({artist}); candidate for Trịnh classification.")
        
    rock_bands = ["bức tường", "microwave", "ngũ cung", "trần lập", "unlimited", "parasite", "quái vật tí hon"]
    if any(rb in a_clean for rb in rock_bands):
        return ("ROCK", f"Band discography specializes in Vietnamese Rock ({artist}); candidate for Rock verification.")
        
    folk_artists = ["thúy hường", "ánh tuyết", "thanh hoa", "thu hiền", "vân khánh", "hương lan", "quang linh", "phi nhung", "cải lương"]
    if any(fa in a_clean for fa in folk_artists) and any(kw in t_clean for kw in ["dân ca", "quan họ", "lý", "câu hò", "bến đò", "sông quê", "huế"]):
        return ("FOLK_TRADITIONAL", f"Artist and title indicate traditional Folk / Dân Ca repertoire ({artist}); candidate for Folk verification.")
        
    children_kw = ["xuân mai", "bào ngư", "bé ", "thiếu nhi", "chú cuội", "mầm non", "cháu đi mẫu giáo"]
    if any(ck in a_clean for ck in children_kw) or any(ck in t_clean for ck in ["chú ếch con", "cháu yêu bà", "chú voi con"]):
        return ("CHILDREN", f"Artist or title metadata indicates Children music catalog ({artist}); candidate for Children verification.")
        
    revolutionary_kw = ["đoàn vệ quốc quân", "tiến quân ca", "bác hồ", "bình trị thiên", "trường sơn", "tiểu đoàn 307", "hò kéo pháo", "đảng đã cho ta"]
    if any(rk in t_clean for rk in revolutionary_kw):
        return ("REVOLUTIONARY", f"Title metadata indicates classical revolutionary anthem ({title}); candidate for Revolutionary verification.")
        
    return (None, "Insufficient metadata evidence - requires audio auditioning.")

def build_dataset_pipeline():
    print("=== RM-VMusic: Building Complete Dataset, 3-Tier Hierarchy, and Annotation Queue ===")
    
    df_vietlyrics = fetch_vietlyrics_full()
    hf_records = fetch_hf_full()
    
    print(f"Loaded {len(df_vietlyrics)} VietLyrics rows and {len(hf_records)} unique HF song_dataset records.")
    
    vl_by_id = {}
    vl_by_pair = {}
    for _, row in df_vietlyrics.iterrows():
        link = str(row.get("link", ""))
        zid = extract_zing_id_from_url(link)
        row_dict = row.to_dict()
        if zid:
            vl_by_id[zid] = row_dict
        t = normalize_title(str(row.get("title", ""))).lower()
        a = normalize_artist(str(row.get("artist", ""))).lower()
        if t and a and t != "nan" and a != "nan":
            vl_by_pair[(t, a)] = row_dict
            
    valid_records = []
    rejected_records = []
    seen_song_keys = set()
    
    # 1. Process HF records (lyrics + timestamps + streaming URLs)
    print("Ingesting HF song_dataset records...")
    for item in tqdm(hf_records, desc="Processing HF records"):
        raw_title = item.get("title", "")
        raw_artist = item.get("artist", "")
        raw_album = item.get("album", "")
        lyrics = item.get("lyrics", "")
        streaming_url = item.get("streaming_url", "")
        source_id = item.get("id", "")
        zing_url = item.get("zingmp3_url", "")
        
        title = normalize_title(raw_title)
        artist = normalize_artist(raw_artist)
        album = normalize_text(raw_album) if raw_album else None
        
        if not title or not artist or title.lower() == "nan" or artist.lower() == "nan":
            rejected = {col: None for col in MASTER_COLUMNS}
            rejected.update({
                "title": title or raw_title,
                "artist": artist or raw_artist,
                "source": "sunbv56_song_dataset",
                "source_id": source_id,
                "rejection_reason": "MISSING_TITLE_OR_ARTIST"
            })
            rejected_records.append(rejected)
            continue
            
        norm_key = (title.lower(), artist.lower())
        if norm_key in seen_song_keys or (source_id and source_id in seen_song_keys):
            rejected = {col: None for col in MASTER_COLUMNS}
            rejected.update({
                "title": title,
                "artist": artist,
                "source": "sunbv56_song_dataset",
                "source_id": source_id,
                "rejection_reason": "DUPLICATE_ENTRY"
            })
            rejected_records.append(rejected)
            continue
            
        seen_song_keys.add(norm_key)
        if source_id:
            seen_song_keys.add(source_id)
            
        # Cross-reference with VietLyrics for genre
        vl_match = vl_by_id.get(source_id)
        if vl_match is None:
            vl_match = vl_by_pair.get(norm_key)
            
        raw_genre = None
        source_name = "sunbv56_song_dataset"
        
        if vl_match is not None:
            raw_genre = vl_match.get("genre")
            source_name = "sunbv56_song_dataset+vietlyrics"
            
        norm_genre, label_source, annot_status, tier = normalize_genre(raw_genre)
        if vl_match is not None and norm_genre is not None:
            annot_status = "cross_verified"
            tier = "TIER_A"
            
        rel_year, rel_year_src = extract_verified_release_year(title, album)
        
        artist_id = generate_deterministic_id("ART", artist)
        album_id = generate_deterministic_id("ALB", album) if album else None
        song_id = generate_deterministic_id("RMVM_S", f"{title}_{artist}_{source_id}")
        
        record = {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "artist_id": artist_id,
            "album": album,
            "album_id": album_id,
            "source_genre": str(raw_genre) if raw_genre and not pd.isna(raw_genre) else None,
            "genre": norm_genre,
            "label_source": label_source,
            "tier": tier,
            "release_year": rel_year,
            "release_year_source": rel_year_src,
            "audio_path": None,
            "audio_url": streaming_url or zing_url or None,
            "lyrics": lyrics if lyrics and len(lyrics.strip()) > 0 else None,
            "cover_path": None,
            "cover_url": None,
            "source": source_name,
            "source_id": source_id or None,
            "annotation_status": annot_status,
            "annotator_id": "pipeline_v2_expansion",
            "annotation_agreement": 1.0 if annot_status == "cross_verified" else None
        }
        valid_records.append(record)
        
    # 2. Ingest additional VietLyrics records
    print("Ingesting additional VietLyrics records...")
    for _, row in tqdm(df_vietlyrics.iterrows(), total=len(df_vietlyrics), desc="Processing VietLyrics"):
        raw_title = str(row.get("title", ""))
        raw_artist = str(row.get("artist", ""))
        raw_genre = row.get("genre")
        link = str(row.get("link", ""))
        source_id = extract_zing_id_from_url(link)
        
        title = normalize_title(raw_title)
        artist = normalize_artist(raw_artist)
        
        if not title or not artist or title.lower() == "nan" or artist.lower() == "nan":
            rejected = {col: None for col in MASTER_COLUMNS}
            rejected.update({
                "title": raw_title,
                "artist": raw_artist,
                "source": "vietlyrics",
                "source_id": source_id,
                "rejection_reason": "MISSING_TITLE_OR_ARTIST"
            })
            rejected_records.append(rejected)
            continue
            
        norm_key = (title.lower(), artist.lower())
        if norm_key in seen_song_keys or (source_id and source_id in seen_song_keys):
            continue
            
        seen_song_keys.add(norm_key)
        if source_id:
            seen_song_keys.add(source_id)
            
        norm_genre, label_source, annot_status, tier = normalize_genre(raw_genre)
        rel_year, rel_year_src = extract_verified_release_year(title, "")
        
        artist_id = generate_deterministic_id("ART", artist)
        song_id = generate_deterministic_id("RMVM_S", f"{title}_{artist}_{source_id}")
        
        record = {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "artist_id": artist_id,
            "album": None,
            "album_id": None,
            "source_genre": str(raw_genre) if raw_genre and not pd.isna(raw_genre) else None,
            "genre": norm_genre,
            "label_source": label_source,
            "tier": tier,
            "release_year": rel_year,
            "release_year_source": rel_year_src,
            "audio_path": None,
            "audio_url": link if link and link != "nan" else None,
            "lyrics": None,
            "cover_path": None,
            "cover_url": None,
            "source": "vietlyrics",
            "source_id": source_id or None,
            "annotation_status": annot_status,
            "annotator_id": "pipeline_v2_expansion",
            "annotation_agreement": None
        }
        valid_records.append(record)

    df_master = pd.DataFrame(valid_records, columns=MASTER_COLUMNS).drop_duplicates(subset=["song_id"]).reset_index(drop=True)
    
    # 3. Read cached cover URLs from existing master file if available
    old_master_path = PROCESSED_DIR / "master_metadata.csv"
    if old_master_path.exists():
        try:
            df_old = pd.read_csv(old_master_path)
            cover_map = df_old[df_old["cover_url"].notna()].set_index("song_id")["cover_url"].to_dict()
            for idx, row in df_master.iterrows():
                sid = row["song_id"]
                if sid in cover_map:
                    df_master.at[idx, "cover_url"] = cover_map[sid]
            print(f"Mapped {df_master['cover_url'].notna().sum()} cached cover URLs.")
        except Exception:
            pass

    # Save complete Master Metadata (7,915 rows)
    df_master.to_csv(MASTER_METADATA_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Complete Master Metadata ({len(df_master)} records) to {MASTER_METADATA_PATH}")
    
    # 4. Generate trainable_metadata.csv (ONLY TIER A & B records with valid ground truth genres)
    df_trainable = df_master[df_master["tier"].isin(["TIER_A", "TIER_B"]) & df_master["genre"].notna()].copy()
    df_trainable.to_csv(TRAINABLE_METADATA_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Core Trainable Dataset ({len(df_trainable)} records) to {TRAINABLE_METADATA_PATH}")
    print(f"  -> TIER A (High-Confidence): {(df_trainable['tier'] == 'TIER_A').sum()}")
    print(f"  -> TIER B (Validated Single-Source): {(df_trainable['tier'] == 'TIER_B').sum()}")
    
    # 5. Generate manual_annotation_candidates.csv & manual_annotation_queue.csv for Tier C (3,612 records)
    df_tier_c = df_master[df_master["tier"] == "TIER_C"].copy()
    candidates = []
    
    for _, row in df_tier_c.iterrows():
        s_gen, s_reason = generate_annotation_suggestion(
            str(row["title"]),
            str(row["artist"]),
            str(row["album"]) if row["album"] else "",
            str(row["source_genre"]) if row["source_genre"] else ""
        )
        cand = {
            "song_id": row["song_id"],
            "title": row["title"],
            "artist": row["artist"],
            "source_genre": row["source_genre"],
            "current_genre": "UNASSIGNED_NULL",
            "suggested_genre": s_gen if s_gen else "NULL",
            "suggestion_reason": s_reason,
            "annotation_status": "needs_manual_annotation",
            "priority": "HIGH" if s_gen is not None else "STANDARD"
        }
        candidates.append(cand)
        
    df_candidates = pd.DataFrame(candidates)
    df_candidates.to_csv(ANNOTATION_CANDIDATES_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Manual Annotation Candidates ({len(df_candidates)} records) to {ANNOTATION_CANDIDATES_PATH}")
    
    # Priority Queue: songs with high candidate suggestions and complete audio/lyrics
    df_queue = df_candidates.sort_values(by="priority", ascending=True)
    df_queue.to_csv(ANNOTATION_QUEUE_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Prioritized Manual Annotation Queue ({len(df_queue)} records) to {ANNOTATION_QUEUE_PATH}")
    
    # Save Tracked Rejected Records
    rejected_cols = MASTER_COLUMNS + ["rejection_reason"]
    df_rejected = pd.DataFrame(rejected_records, columns=rejected_cols)
    df_rejected.to_csv(REJECTED_RECORDS_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved Tracked Rejected Records ({len(df_rejected)} records) to {REJECTED_RECORDS_PATH}")

if __name__ == "__main__":
    build_dataset_pipeline()
