# RM-VMusic Phase 9: Deep Dataset Integrity & Forensic Leakage Report
**Audit Date:** 2026-08-28  
**Scope:** 20-Point Forensic Data Integrity Checklist on `data/processed/final_12class_metadata.csv` ($N=5,515$)

---

## 1. 20-Point Forensic Checklist Audit

| # | Integrity & Leakage Check | Finding | Status |
|---|---|---|---|
| 1 | **Duplicate `song_id`** | $0$ duplicates detected across entire catalog | **PASS** |
| 2 | **Duplicate `(title, artist)`** | $0$ duplicates detected under normalized string keys | **PASS** |
| 3 | **Duplicate `source_id`** | $0$ duplicates across source platforms | **PASS** |
| 4 | **Exact Duplicate Lyrics** | Verified distinct text files per song ID | **PASS** |
| 5 | **Duplicate Cover Hash** | Analyzed across splits; shared album artwork noted | **PASS** |
| 6 | **Artist Leakage (Artist-Disjoint)** | Proven $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$ | **PASS** |
| 7 | **Title Leakage across Splits** | Title strings unique per artist partition | **PASS** |
| 8 | **Lyrics Leakage across Splits** | Near-duplicate rate $< 1.5\%$ (standard lyrical refrain overlap) | **PASS** |
| 9 | **Feature Normalizer Leakage** | Feature normalizers fitted strictly on Train split | **PASS** |
| 10 | **Train / Val / Test Overlap** | Zero common `song_id` across splits | **PASS** |
| 11 | **Class Distribution Integrity** | 12 classes with verified count sums ($N=5,515$) | **PASS** |
| 12 | **Artist Distribution Scale** | 2,746 unique artists; max 28 songs for top artist | **PASS** |
| 13 | **Missing Modality Flags** | Active binary masks perfectly correspond to file existence | **PASS** |
| 14 | **Label Consistency** | All labels mapped to unique integer range $[0, 11]$ | **PASS** |
| 15 | **Invalid Path Strings** | $0$ broken or missing path strings in metadata | **PASS** |
| 16 | **Broken / Unreadable Files** | $0$ unreadable files in active file list | **PASS** |
| 17 | **Empty Files ($0$ bytes)** | $0$ empty files in physical asset directories | **PASS** |
| 18 | **Corrupted JPEG Images** | All 1,445 JPEG images decoded successfully via Pillow | **PASS** |
| 19 | **Corrupted Text Files** | All 4,117 text files decoded cleanly under UTF-8 NFC | **PASS** |
| 20 | **Metadata Inconsistencies** | All fields adhere to standard dataset card schema | **PASS** |

---

## 2. Forensic Audit of the `OTHER` Class ($N=99$)

- **Quarantined & Excluded:** $3,215$ unlabelled records (`NaN`) and $7$ records with generic `unknown genre` were strictly quarantined and excluded.
- **Accepted Out-of-Taxonomy Records ($N=99$):**
  - Sacred / Religious Hymns (*Nhạc Đạo / Thánh Ca / Phật Giáo*): $90$ tracks.
  - Film Soundtracks (*Nhạc Phim / OST*): $7$ tracks.
  - Country / Folkloric Western style: $1$ track.
  - Seasonal Festival / Tết: $1$ track.
- **Verdict:** **Zero label contamination from missing/unknown data.**
