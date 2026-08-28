# RM-VMusic Phase 12: External Dataset Discovery & Legitimate Data Acquisition Report
**Evaluation Date:** 2026-08-28

---

## 1. External Dataset Landscape & Usability Audit

| Dataset Name | Platform | Stated License | Sample Count | Score (/100) | Reviewer Determination |
|---|---|---|---|---|---|
| **VietLyrics (`vi-song-7k-public`)** | Hugging Face / arXiv | CC-BY-NC-SA 4.0 | 8,428 | **75 / 100** | **USABLE (Strict Academic License)** |
| **sunbv56 (`song_dataset`)** | Hugging Face | Open Academic | 9,344 | **56 / 100** | **USABLE (Lyrics & Timestamps Only)** |
| **Vietnam Traditional Music (VNTM)**| Kaggle | CC0 / Public Domain | 1,250 | **70 / 100** | **USABLE FOR FOLK_TRADITIONAL** |
| **Vietnamese Music Dataset** | Hugging Face | Unspecified | 450 | **42 / 100** | **REJECTED (Unclear License)** |
| **Zing MP3 Public Stream Index** | Commercial Streaming | Proprietary | > 100,000 | **47 / 100** | **REJECTED (Copyright / DRM Boundary)** |

---

## 2. Temporal & Class Gap Realities

1. **Why `NHAC_TRINH` and `CHILDREN` cannot be artificially increased for $\ge 2021$:**
   - No open-license external dataset currently indexes post-2021 releases of *Nhạc Trịnh Công Sơn* or modern children's nursery recordings with verified release year tags.
   - In accordance with Phase 12 scientific rules, **zero fake release years were created**.
2. **Expansion Candidates Identified:**
   - **54 verified tracks** from VietLyrics ground truth integrated into candidate catalog.
   - **V3 Candidate Catalog:** **5,569 tracks** across 2,770 unique artists.
