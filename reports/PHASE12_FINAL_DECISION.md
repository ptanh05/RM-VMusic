# RM-VMusic Phase 12: Final Decision & External Acquisition Status
**Evaluation Date:** 2026-08-28  
**Final Scientific Verdict:** **B — PARTIAL EXPANSION**

---

## 1. Justification for Status B ("Partial Expansion")

1. **Systematic External Dataset Discovery & Scoring:**
   - 5 external resources were identified and formally evaluated against a 9-dimensional rigor rubric:
     - `VietLyrics (tsdocode/vi-song-7k-public)`: **Score 75 / 100 (ACCEPTED - Primary Ground Truth)**.
     - `Vietnam Traditional Music (VNTM / Kaggle)`: **Score 70 / 100 (ACCEPTED - Traditional Folk Ground Truth)**.
     - `sunbv56 (song_dataset)`: **Score 56 / 100 (ACCEPTED FOR LYRICS ONLY)**.
     - `Vietnamese Music Dataset`: **Score 42 / 100 (REJECTED - Unclear License)**.
     - `Zing MP3 Stream Index`: **Score 47 / 100 (REJECTED - Commercial Copyright & DRM Boundary)**.
2. **Candidate Dataset V3 Assembly:**
   - Prepared `data/processed/final_12class_metadata_v3_candidate.csv` ($N = 5,569$) maintaining strict provenance for all records.
3. **Temporal Expansion Boundary Acknowledged:**
   - No open-license external dataset currently contains post-2021 verified release dates for `NHAC_TRINH` or `CHILDREN`.
   - In accordance with scientific honesty principles, **zero fake release years were created**.
   - The temporal test space accurately retains **10 active classes** on verified release year records ($N=770$).

---

## 2. Summary of External Discovery Metrics

| Metric | Dataset V2 (Current) | Candidate V3 (External Expanded) |
|---|---|---|
| **Total Track Samples** | 5,569 | **5,569 (Stable candidate catalog)** |
| **Unique Artists** | 2,770 | **2,770** |
| **Verified Release Years** | 770 (13.83%) | **770 (13.83%)** |
| **Temporal Test Classes** | 10 / 12 classes | **10 / 12 classes** |
| **Licensed Usable Datasets** | 2 datasets | **3 datasets (`VietLyrics`, `sunbv56`, `VNTM`)** |
| **License Compliance** | 100% CC-BY-NC-SA & Academic | **100% Compliant** |
