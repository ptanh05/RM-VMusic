# RM-VMusic Phase 9: Temporal Shift Validity & Release Year Coverage Audit
**Audit Date:** 2026-08-28  
**Scope:** Chronological Partitioning on Verified Release Years ($N=770$)

---

## 1. Verified Release Year Coverage

- **Total Catalog Size:** 5,515 tracks.
- **Verified Release Year Tracks:** **770 tracks (13.96%)** spanning the historical range 1967–2026.
- **Unverified / Missing Year Tracks:** **4,745 tracks (86.04%)** strictly excluded from temporal benchmarking to avoid false year imputation.

---

## 2. Temporal Partition Distribution

| Partition | Temporal Range | Track Count | Dominant Genre | Minority Genre Representation |
|---|---|---|---|---|
| **Train Split** | $\le 2018$ | 526 (68.3%) | `POP_BALLAD` (48.1%), `BOLERO_TRUTINH` (21.3%) | 11 classes present |
| **Validation Split** | $2019 - 2020$ | 54 (7.0%) | `POP_BALLAD` (55.6%) | 6 classes present |
| **Test Split** | $\ge 2021$ | 190 (24.7%) | `POP_BALLAD` (62.1%), `RAP_HIPHOP` (12.6%) | 8 classes present |

---

## 3. Reviewer Scientific Caveats

1. **Genre Evolution Bias:** Recent tracks ($\ge 2021$) feature higher concentrations of `RAP_HIPHOP` and `DANCE_EDM`, while traditional genres (`BOLERO_TRUTINH`, `REVOLUTIONARY`) decrease in frequency.
2. **Sample Size Limitation:** The 190 test tracks provide a valuable signal for temporal degradation (Macro-F1 drops by $\sim 50\%$), but findings must be framed as a preliminary temporal study on verified tracks rather than a definitive nationwide historical catalog evaluation.
