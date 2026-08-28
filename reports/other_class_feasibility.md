# RM-VMusic Phase 7B: OTHER Class Feasibility & Semantic Audit Report
**Audit Date:** 2026-08-28  
**Scope:** Semantic Analysis of Tier C & Out-of-Taxonomy Records ($N=3,322$)  
**Outcome:** Formal 12th Class Feasibility Verified with Strict Evidence Filtering

---

## 1. Executive Summary & Semantic Definition

In classical music genre classification and distribution shift research, an **`OTHER`** class serves to capture legitimate out-of-distribution or non-target music genres. However, in scientific benchmark construction:
- **`OTHER` must possess explicit semantic evidence** (e.g. Religious music, Film Soundtracks/OST, Holiday/Festival, Country).
- **`OTHER` must NEVER become a dumping ground for unlabelled, ambiguous, or missing data** (`UNKNOWN`, `UNLABELED`, `NaN`).

---

## 2. Exhaustive Audit of Tier C Candidates ($N=3,322$)

Across the 3,322 unmapped records in `master_metadata.csv`:

| Raw Source Genre String | Track Count | Semantic Description | Feasibility Decision for `OTHER` |
|---|---|---|---|
| `NaN` (Missing Source Tag) | 3,215 | VietLyrics tracks without upstream genre | **REJECTED FROM OTHER** (Retained in `manual_annotation_queue.csv`) |
| `unknown genre` | 7 | Upstream unlabelled metadata | **REJECTED FROM OTHER** |
| `nhạc tôn giáo` | 87 | Vietnamese Christian & Buddhist sacred hymns | **ACCEPTED INTO OTHER** (Strong semantic evidence) |
| `nhạc phim` | 7 | Film OST / Cinematic compositions | **ACCEPTED INTO OTHER** (Distinct functional genre) |
| `nhạc đạo` | 3 | Vietnamese spiritual choral music | **ACCEPTED INTO OTHER** (Sacred genre) |
| `tết` | 1 | Traditional New Year holiday song | **ACCEPTED INTO OTHER** (Functional seasonal music) |
| `country` | 1 | Vietnamese Country acoustic arrangement | **ACCEPTED INTO OTHER** |
| `âu mỹ` | 1 | Western non-target track | **REJECTED** (Out of Vietnamese language scope) |

---

## 3. Formally Accepted `OTHER` Class Profile ($N=98$)

- **Total Validated Samples:** **98 tracks** (1.78% of the 5,514 total 12-class dataset).
- **Artist Diversity:** **53 unique artists** (average 1.85 tracks per artist).
- **Quality Tier:** Assigned to **`TIER_B`** with `label_source = "curated_out_of_taxonomy"` and `label_confidence = 0.90`.
- **Modality Availability:**
  - Physical Lyrics: 85 / 98 (86.73%)
  - Physical Covers: 12 / 98 (12.24%)
  - Physical Audio: 0 / 98 (0.00%)
- **Comparable Class Size:** `OTHER` ($N=98$) is balanced with `CHILDREN` ($N=93$, 1.69%) and `RB_SOUL` ($N=132$, 2.39%).

---

## 4. Final 12-Class Taxonomy Specification

1. `POP_BALLAD` (3,031 tracks, 54.97%)
2. `BOLERO_TRUTINH` (807 tracks, 14.64%)
3. `INSTRUMENTAL` (287 tracks, 5.20%)
4. `RAP_HIPHOP` (221 tracks, 4.01%)
5. `FOLK_TRADITIONAL` (200 tracks, 3.63%)
6. `DANCE_EDM` (193 tracks, 3.50%)
7. `REVOLUTIONARY` (170 tracks, 3.08%)
8. `NHAC_TRINH` (145 tracks, 2.63%)
9. `ROCK` (137 tracks, 2.48%)
10. `RB_SOUL` (132 tracks, 2.39%)
11. `OTHER` (**98 tracks**, 1.78%)
12. `CHILDREN` (93 tracks, 1.69%)

**Total 12-Class Dataset:** **5,514 tracks** (2,746 unique artists).
