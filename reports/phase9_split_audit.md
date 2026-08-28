# RM-VMusic Phase 9: Benchmark Split & Preprocessing Isolation Audit
**Audit Date:** 2026-08-28  
**Scope:** Verification of All 5 Benchmark Splits and Absence of Test Set Contamination

---

## 1. Benchmark Partition Summary Table

| Benchmark Split | Train Partition | Validation Partition | Test Partition | Total Tracks | Isolation Guarantee |
|---|---|---|---|---|---|
| **IID Split** | 3,860 (70.0%) | 827 (15.0%) | 828 (15.0%) | 5,515 | Stratified random split |
| **Artist Disjoint** | 3,860 (1,908 artists) | 827 (428 artists) | 828 (411 artists) | 5,515 | **Strict 0% Artist Overlap** |
| **Temporal Shift** | 526 ($\le 2018$) | 54 ($2019-2020$) | 190 ($\ge 2021$) | 770 | Verified metadata years |
| **Label Shift** | 3,996 (72.5%) | 584 (10.6%) | 935 (16.9%) | 5,515 | Controlled class prior shift |
| **Missing Modality** | — | — | 828 (Test) / 5,515 | 5,515 | Modality dropout stress-test |

---

## 2. Preprocessing & Vocabulary Leakage Audit

1. **TF-IDF Vocabulary Fitting:**
   - Vectorizer is fitted strictly on the $2,877$ valid text files in `final12_iid_train.csv`.
   - Test and Validation partitions are strictly transformed out-of-sample via `vectorizer.transform()`.
   - **Leakage Finding:** **PASSED (Zero vocabulary contamination)**.
2. **Cover Image Normalization:**
   - Features are computed per-sample directly from image pixel arrays; no global statistics are fitted across splits.
   - **Leakage Finding:** **PASSED**.
3. **Class Weighting:**
   - Class weights for the loss function are computed exclusively from the `Train` split labels.
   - **Leakage Finding:** **PASSED**.
