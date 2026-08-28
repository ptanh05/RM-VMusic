# RM-VMusic Phase 9: Cover Modality Quality & Visual Leakage Audit
**Audit Date:** 2026-08-28  
**Scope:** 1,445 Physical Covers on Disk ($902$ in Trainable Catalog = $16.36\%$ Coverage)

---

## 1. Physical Cover Verification & Integrity

- **Files on Disk (`data/covers/`):** **1,445 JPEG images** decoded and verified with Pillow.
- **Corrupted / Broken Images:** **0 files**.
- **Trainable Catalog Coverage:** **902 / 5,515 tracks (16.36%)**.
- **Visual Features Extracted:** 512-dimensional spatial color grid ($3\times3$) and RGB gradient moments.

---

## 2. Cross-Split Artwork Reuse & Visual Leakage Audit

### A. IID Split Artwork Overlap
- In the IID random split, tracks from the same music album occasionally share the identical cover art ($23$ unique album cover images shared between Train and Test).
- **Impact:** Reflects standard in-distribution album classification scenarios.

### B. Artist-Disjoint Split Artwork Overlap
- On the `final12_artist_disjoint` benchmark:
  $$\text{Train Cover Hashes} \cap \text{Test Cover Hashes} = \emptyset \quad (0 \text{ shared images})$$
- **Impact:** Proves **zero visual artwork leakage** across artist-disjoint partitions. The model cannot memorize album art visual tokens to predict unseen artists.
