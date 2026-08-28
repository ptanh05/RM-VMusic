# RM-VMusic Phase 8: Deep Dataset & Taxonomy Scientific Audit Report
**Audit Standard:** Strict ML Conference Reviewer Protocol (No Pseudo-Features, Zero Fabricated Data)  
**Audit Date:** 2026-08-28  
**Audited Target:** `data/processed/final_12class_metadata.csv` ($N=5,515$)

---

## 1. Catalog Scale & Class Distribution Audit

| Class Index | Genre Class | Sample Count | Class % | Unique Artists | Physical Lyrics | Physical Covers | Physical Audio | Imbalance Ratio vs Min |
|---|---|---|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | 2,726 | 587 | 0 | 32.59x |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | 694 | 167 | 0 | 8.68x |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | 217 | 44 | 0 | 3.09x |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | 111 | 21 | 0 | 2.38x |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | 82 | 18 | 0 | 2.15x |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | 149 | 21 | 0 | 2.08x |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | 23 | 4 | 0 | 1.83x |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | 12 | 2 | 0 | 1.56x |
| 8 | `ROCK` | 137 | 2.48% | 20 | 15 | 6 | 0 | 1.47x |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | 14 | 4 | 0 | 1.42x |
| 10 | `OTHER` | 99 | 1.80% | 54 | 0 | 14 | 0 | 1.06x |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | 74 | 14 | 0 | 1.00x |
| **Total** | **12 Classes** | **5,515** | **100.00%** | **2,746** | **4,117** | **902** | **0** | **Gini: 0.6102** |

---

## 2. OTHER Class Semantic Evidence & Contamination Audit

- **Candidate Pool Analyzed:** 3,322 Tier C master records.
- **Contamination Filtering Policy:**
  - **3,215 records with missing genre tags (`NaN`)** were strictly quarantined and excluded.
  - **7 records with generic `unknown genre`** tags were strictly quarantined and excluded.
- **Positive Out-of-Taxonomy Records Accepted ($N=99$):**
  - *Nhạc Tôn giáo / Thánh ca / Đạo ca (`nhạc tôn giáo`, `nhạc đạo`)*: 90 tracks across 48 artists.
  - *Nhạc phim / Original Soundtrack (`nhạc phim`, `ost`, `soundtrack`)*: 7 tracks across 5 artists.
  - *Nhạc Country / Đồng quê Việt*: 1 track (`country`).
  - *Nhạc Lễ hội / Tết*: 1 track (`tết`).
- **Audit Finding:** **PASS (Zero contamination from unlabeled data)**.

---

## 3. Catalog-Level Deduplication Audit

- Exact `song_id` duplicates: **0**
- Exact `(title, artist)` duplicates: **0**
- Exact `source_id` duplicates: **0**
- Unicode NFC normalization check: **0 encoding errors**
- **Audit Finding:** **PASS (Zero accidental duplicates)**.

---

## 4. Benchmark Split Isolation & Zero Artist Leakage Verification

| Benchmark Split | Train / Val / Test Sizes | Song Overlap (Tr∩Va / Tr∩Te / Va∩Te) | Artist Overlap (Tr∩Va / Tr∩Te / Va∩Te) | Verification Status |
|---|---|---|---|---|
| **IID** | 3,860 / 827 / 828 | 0 / 0 / 0 | 321 / 324 / 166 | **PASS (Standard IID)** |
| **Artist Disjoint** | 3,860 / 827 / 828 | **0 / 0 / 0** | **0 / 0 / 0** | **PASS (STRICT 0% LEAKAGE)** |
| **Temporal Shift** | 526 / 54 / 190 | 0 / 0 / 0 | 15 / 22 / 18 | **PASS (Verified Release Years)** |
| **Label Shift** | 3,996 / 584 / 935 | 0 / 0 / 0 | 254 / 332 / 148 | **PASS (Controlled Shift)** |

**Mathematical Proof on Artist Disjoint:**
$$\text{Train Artists } (N=1,908) \cap \text{Val Artists } (N=428) = \emptyset$$
$$\text{Train Artists } (N=1,908) \cap \text{Test Artists } (N=411) = \emptyset$$
$$\text{Val Artists } (N=428) \cap \text{Test Artists } (N=411) = \emptyset$$
