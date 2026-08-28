# RM-VMusic Dataset Card (v2.0 — Phase 7B Physical Release)

## 1. Dataset Summary

**RM-VMusic** (*Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift*) is a curated benchmark for studying distribution shifts and multimodal robustness in Vietnamese music information retrieval.

- **Total Tracks (12-Class Catalog):** 5,515 tracks
- **Master Metadata Catalog:** 8,738 tracks
- **Unique Artists:** 2,746 artists
- **Total Physical Assets on Disk:**
  - **Lyrics:** 4,117 text files (`.txt`)
  - **Covers:** 1,445 JPEG images (`.jpg`)
  - **Audio:** 0 physical files (explicit zero-masking for missing modality)
- **Supported Shifts:** IID, Artist-Disjoint (0% leakage), Temporal (1967–2026), Label Shift, Missing Modality.

---

## 2. Genre Class Distribution

| Index | Genre Class | Track Count | Percentage | Unique Artists | Physical Lyrics | Physical Covers |
|---|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | 2,726 | 587 |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | 694 | 167 |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | 217 | 44 |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | 111 | 21 |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | 82 | 18 |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | 149 | 21 |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | 23 | 4 |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | 12 | 2 |
| 8 | `ROCK` | 137 | 2.48% | 20 | 15 | 6 |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | 14 | 4 |
| 10 | `OTHER` | 99 | 1.80% | 54 | 0 | 14 |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | 74 | 14 |
| **Total** | **12 Classes** | **5,515** | **100.00%** | **2,746** | **4,117** | **902 (Trainable)** |

---

## 3. Provenance & Quality Tiers

- **TIER A (75.4%):** Cross-verified across independent Vietnamese lyric and streaming databases with 100% agreement.
- **TIER B (24.6%):** Curated exact/normalized genre tags from verified sources.
- **Deduplication:** 0 duplicates across `song_id`, `(title, artist)`, or `source_id`.

---

## 4. Benchmark Partitions (`data/splits/`)

- `final12_iid_*.csv`: Stratified random partition (70/15/15).
- `final12_artist_disjoint_*.csv`: Strict artist group-disjoint partition (0% artist leakage).
- `final12_temporal_*.csv`: Verified release year partition (Train $\le 2018$, Val $2019-2020$, Test $\ge 2021$).
- `final12_label_shift_*.csv`: Controlled prior probability shift.
- `final12_missing_modality.csv`: Multimodal availability evaluation split.
