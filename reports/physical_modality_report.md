# RM-VMusic Phase 6: Real Physical Modality Matrix & Verification Report

This document audits the **actual physical assets on disk** (`data/audio/`, `data/lyrics/`, `data/covers/`) versus the **metadata annotations** in `data/processed/trainable_metadata.csv` for all **5,416** trainable records.

---

## 1. Metadata Availability vs Physical Asset Availability

| Modality Dimension | Metadata Available Count | Metadata Coverage (%) | **Physical File Valid Count** | **Physical File Coverage (%)** | Discrepancy / Gap |
|--------------------|--------------------------|-----------------------|-------------------------------|--------------------------------|-------------------|
| **Audio Modality** | 5,401 | 99.72% | **0** | **0.00%** | -5,401 (Expired CDN URLs) |
| **Lyrics Modality** | 4,117 | 76.02% | **4,117** | **76.02%** | **0 (100% Materialized to .txt)** |
| **Cover Artwork** | 888 | 16.40% | **413** | **7.63%** | -475 (404 / Blocked Images) |

> [!WARNING]
> **CRITICAL DISCOVERY ON AUDIO**:
> In prior phases, the dataset reported `99.72% audio coverage` based purely on metadata `audio_url` strings.
> However, physical asset auditing reveals that the 4,406 Zing MP3 streaming URLs (`a128-z3.zmdcdn.me`) use temporary time-limited authorization tokens that return **HTTP 403 Forbidden** when downloaded today, and MusicBrainz recording links are web entity pages.
> **Actual Physical Audio Files on Disk = 0 (0.00%)**.
> **Current dataset is metadata-rich but physically incomplete.**

---

## 2. Genre × Physical Modality Breakdown Matrix

| Genre | Total ($N$) | Physical Audio | Physical Lyrics | Physical Cover | Full Multimodal | Lyrics + Cover | Lyrics Only | Cover Only | Missing All Physical | Physical Multimodal (%) |
|-------|-------------|----------------|-----------------|----------------|-----------------|----------------|-------------|------------|----------------------|--------------------------|
| `POP_BALLAD` | 3031 | 0 (0.0%) | 2726 (89.94%) | 265 (8.74%) | **0** | 74 | 2652 | 191 | 114 | **0.0%** |
| `BOLERO_TRUTINH` | 807 | 0 (0.0%) | 694 (86.0%) | 89 (11.03%) | **0** | 13 | 681 | 76 | 37 | **0.0%** |
| `INSTRUMENTAL` | 287 | 0 (0.0%) | 217 (75.61%) | 27 (9.41%) | **0** | 5 | 212 | 22 | 48 | **0.0%** |
| `RAP_HIPHOP` | 221 | 0 (0.0%) | 111 (50.23%) | 8 (3.62%) | **0** | 4 | 107 | 4 | 106 | **0.0%** |
| `FOLK_TRADITIONAL` | 200 | 0 (0.0%) | 82 (41.0%) | 10 (5.0%) | **0** | 1 | 81 | 9 | 109 | **0.0%** |
| `DANCE_EDM` | 193 | 0 (0.0%) | 149 (77.2%) | 3 (1.55%) | **0** | 1 | 148 | 2 | 42 | **0.0%** |
| `REVOLUTIONARY` | 170 | 0 (0.0%) | 23 (13.53%) | 1 (0.59%) | **0** | 0 | 23 | 1 | 146 | **0.0%** |
| `NHAC_TRINH` | 145 | 0 (0.0%) | 12 (8.28%) | 2 (1.38%) | **0** | 0 | 12 | 2 | 131 | **0.0%** |
| `ROCK` | 137 | 0 (0.0%) | 15 (10.95%) | 3 (2.19%) | **0** | 0 | 15 | 3 | 119 | **0.0%** |
| `RB_SOUL` | 132 | 0 (0.0%) | 14 (10.61%) | 2 (1.52%) | **0** | 1 | 13 | 1 | 117 | **0.0%** |
| `CHILDREN` | 93 | 0 (0.0%) | 74 (79.57%) | 3 (3.23%) | **0** | 0 | 74 | 3 | 16 | **0.0%** |

---

## 3. Overall Physical Modality States Summary

- **Full Multimodal (Audio + Lyrics + Cover)**: **0** (0.00%)
- **Lyrics + Cover (Dual Modality)**: **99** (1.83%)
- **Lyrics Only (Single Modality)**: **4,018** (74.19%)
- **Cover Only (Single Modality)**: **314** (5.80%)
- **Missing All Physical Files**: **985** (18.19%)

---
*Báo cáo kiểm toán tài nguyên vật lý Phase 6 - RM-VMusic Pipeline.*
