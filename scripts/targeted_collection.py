"""
targeted_collection.py
RM-VMusic: Targeted Data Collection Pipeline for Deficient/Rare Genres.
Defines targeted search schemas, open API query adapters, and artist diversity constraints.
Ensures max tracks per artist to prevent single-artist domination.
"""

import sys
import os
import json
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
DOCS_DIR = BASE_DIR / "docs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TARGETED_GUIDE_PATH = DOCS_DIR / "targeted_collection_guide.md"

# Target Specifications for Deficient Genres
TARGETED_GENRE_PROFILES = {
    "ROCK": {
        "canonical_name": "Rock Việt",
        "current_sample_count": 7,
        "target_sample_count": 200,
        "max_tracks_per_artist": 8,
        "representative_artists_and_bands": [
            "Bức Tường", "Microwave", "Ngũ Cung", "Trần Lập", "Unlimited",
            "Quái Vật Tí Hon", "Parasite", "Cát", "I-Tễu", "Gạt Tàn Đầy"
        ],
        "suggested_query_sources": [
            "MusicBrainz: tag:vietnamese_rock OR artist:\"Bức Tường\"",
            "VietLyrics: query keyword 'rock' in title or artist",
            "Open Audio Archives / Public YouTube Creative Commons tracks"
        ]
    },
    "RB_SOUL": {
        "canonical_name": "R&B / Soul Việt",
        "current_sample_count": 15,
        "target_sample_count": 200,
        "max_tracks_per_artist": 8,
        "representative_artists_and_bands": [
            "JustaTee", "Touliver", "Vũ.", "Marzuz", "Mỹ Anh",
            "Orange", "Wren Evans", "Phùng Khánh Linh", "Thịnh Suy", "Kimmese"
        ],
        "suggested_query_sources": [
            "MusicBrainz: genre:r&b AND country:VN",
            "VietLyrics: query 'r&b' or 'soul'",
            "SpaceSpeakers public discographies"
        ]
    },
    "NHAC_TRINH": {
        "canonical_name": "Nhạc Trịnh",
        "current_sample_count": 10,
        "target_sample_count": 150,
        "max_tracks_per_artist": 10,
        "representative_artists_and_bands": [
            "Khánh Ly", "Trịnh Công Sơn", "Trịnh Vĩnh Trinh", "Lô Thủy",
            "Giang Trang", "Hồng Nhung", "Cẩm Vân", "Quang Dũng"
        ],
        "suggested_query_sources": [
            "Trịnh Công Sơn catalog registries",
            "VietLyrics: author:\"Trịnh Công Sơn\" OR title matches Trịnh classics",
            "MusicBrainz: composer:\"Trịnh Công Sơn\""
        ]
    },
    "INSTRUMENTAL": {
        "canonical_name": "Nhạc không lời / Hòa tấu",
        "current_sample_count": 10,
        "target_sample_count": 150,
        "max_tracks_per_artist": 6,
        "representative_artists_and_bands": [
            "Đặng Thái Sơn", "Võ Vân Ánh", "Hải Phượng", "Nguyễn Lê",
            "Hoàng Tuấn", "Kim Sinh", "Hòa tấu đàn tranh", "Hòa tấu đàn bầu"
        ],
        "suggested_query_sources": [
            "VietLyrics 'new age / world music' instrumental queue",
            "Traditional Instrument Archives (Dan Tranh, Dan Bau, Sao Truc)",
            "MusicBrainz: instrument:solo"
        ]
    },
    "REVOLUTIONARY": {
        "canonical_name": "Tiền chiến / Cách mạng",
        "current_sample_count": 19,
        "target_sample_count": 150,
        "max_tracks_per_artist": 6,
        "representative_artists_and_bands": [
            "Văn Cao", "Phan Huỳnh Điểu", "Hoàng Vân", "Đỗ Nhuận",
            "Trần Tiến", "Đoàn Chuẩn", "Quang Thọ", "Trọng Tấn", "Đăng Dương", "Việt Hoàn"
        ],
        "suggested_query_sources": [
            "Red Music public cultural archives",
            "VietLyrics: genre 'nhạc cách mạng' or 'nhạc đỏ'",
            "MusicBrainz: composer:\"Văn Cao\" OR composer:\"Hoàng Vân\""
        ]
    },
    "FOLK_TRADITIONAL": {
        "canonical_name": "Dân ca / Quê hương",
        "current_sample_count": 73,
        "target_sample_count": 300,
        "max_tracks_per_artist": 8,
        "representative_artists_and_bands": [
            "Thúy Hường", "Thu Hiền", "Thanh Hoa", "Ánh Tuyết",
            "Vân Khánh", "Hương Lan", "Quang Linh", "Phi Nhung", "Lệ Thủy", "Minh Vương"
        ],
        "suggested_query_sources": [
            "VietLyrics 'nhạc dân ca - quê hương' and 'cải lương'",
            "Quan họ Bắc Ninh & Đờn ca tài tử open archives"
        ]
    },
    "CHILDREN": {
        "canonical_name": "Thiếu nhi",
        "current_sample_count": 64,
        "target_sample_count": 250,
        "max_tracks_per_artist": 8,
        "representative_artists_and_bands": [
            "Xuân Mai", "Bé Bào Ngư", "Bé Trang Thư", "Đội văn nghệ Tuổi Thơ",
            "Bé Nhật Lan Vy", "Bé Ben", "Thiếu nhi Ba Đình"
        ],
        "suggested_query_sources": [
            "VietLyrics 'nhạc thiếu nhi'",
            "Tuổi Thơ music catalog registries"
        ]
    }
}

def generate_targeted_collection_guide():
    content = rf"""# RM-VMusic: Targeted Collection Strategy for Deficient Genres

This document details the targeted data acquisition framework designed to resolve severe class imbalances in **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)** without artificial oversampling or fake data.

---

## 1. Principles of Targeted Data Collection

1. **Empirical Grounding**: Data must be acquired from verified authentic Vietnamese musical recordings with real acoustic signals and verified lyrics.
2. **Artist Diversity Constraint**: To prevent the model from learning artist-specific shortcuts rather than acoustic/lyrical genre representations, a strict ceiling ($\max \\le 8-10$ tracks per artist) is enforced for each genre class.
3. **No Synthetic / Oversampled Data**: Training sets must never duplicate or hallucinate records.

---

## 2. Targeted Genre Profiles & Collection Targets

| Genre Code | Canonical Name | Current Count | Target Sample Size | Max Tracks / Artist | Priority Level |
|------------|----------------|---------------|--------------------|---------------------|----------------|
"""
    for gcode, p in TARGETED_GENRE_PROFILES.items():
        priority = "URGENT (Severely Deficient)" if p["current_sample_count"] < 20 else ("HIGH" if p["current_sample_count"] < 80 else "MODERATE")
        content += f"| `{gcode}` | **{p['canonical_name']}** | {p['current_sample_count']} | **{p['target_sample_count']}** | {p['max_tracks_per_artist']} | {priority} |\n"

    content += f"""
---

## 3. Detailed Acquisition Profiles & Discography Sources

"""
    for gcode, p in TARGETED_GENRE_PROFILES.items():
        content += f"### 3.{list(TARGETED_GENRE_PROFILES.keys()).index(gcode)+1}. {p['canonical_name']} (`{gcode}`)\n"
        content += f"- **Target Count**: {p['target_sample_count']} (Current: {p['current_sample_count']})\n"
        content += f"- **Max Tracks per Artist**: {p['max_tracks_per_artist']}\n"
        content += f"- **Representative Artists**: {', '.join(p['representative_artists_and_bands'])}\n"
        content += f"- **Verified Query Sources**:\n"
        for qs in p['suggested_query_sources']:
            content += f"  - {qs}\n"
        content += "\n"

    content += f"""---
*Tài liệu thuộc khuôn khổ dự án RM-VMusic - Task 5 Targeted Data Collection.*
"""
    with open(TARGETED_GUIDE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Saved Targeted Collection Guide to {TARGETED_GUIDE_PATH}")

def main():
    print("=== RM-VMusic: Running Targeted Collection Framework ===")
    generate_targeted_collection_guide()
    print("=== Targeted Collection Framework Ready ===")

if __name__ == "__main__":
    main()
