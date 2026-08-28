# RM-VMusic: Comprehensive Dataset & Split Forensic Audit Report
**Audit Date:** 2026-08-28  
**Audit Standard:** Forensic Senior Reviewer Audit (Code & Disk Truth Only)

---

## 1. Dataset & File Inventory (Phase A)

| Dataset Entity | File Path | N Samples | N Artists | N Classes | Verification Status |
|---|---|---|---|---|---|
| **Master Catalog** | `data/processed/master_metadata.csv` | 8,738 | 3,472 | 12 | **EXISTS (PROVEN)** |
| **Trainable Metadata** | `data/processed/final_12class_metadata.csv` | 5,515 | 2,746 | 12 | **EXISTS (PROVEN)** |
| **IID Train** | `data/splits/final12_iid_train.csv` | 3,860 | 2,137 | 12 | **EXISTS (PROVEN)** |
| **IID Validation** | `data/splits/final12_iid_val.csv` | 827 | 631 | 12 | **EXISTS (PROVEN)** |
| **IID Test** | `data/splits/final12_iid_test.csv` | 828 | 642 | 12 | **EXISTS (PROVEN)** |
| **Artist Disjoint Train** | `data/splits/final12_artist_disjoint_train.csv` | 3,860 | 1,908 | 12 | **EXISTS (PROVEN)** |
| **Artist Disjoint Val** | `data/splits/final12_artist_disjoint_val.csv` | 827 | 428 | 12 | **EXISTS (PROVEN)** |
| **Artist Disjoint Test** | `data/splits/final12_artist_disjoint_test.csv` | 828 | 411 | 12 | **EXISTS (PROVEN)** |
| **Temporal Train** | `data/splits/final12_temporal_train.csv` | 526 | 88 | 11 | **EXISTS (PROVEN)** |
| **Temporal Val** | `data/splits/final12_temporal_val.csv` | 54 | 29 | 7 | **EXISTS (PROVEN)** |
| **Temporal Test** | `data/splits/final12_temporal_test.csv` | 190 | 62 | 10 | **EXISTS (PROVEN)** |
| **Label Shift Train** | `data/splits/final12_label_shift_train.csv` | 3,996 | 2,218 | 12 | **EXISTS (PROVEN)** |
| **Label Shift Val** | `data/splits/final12_label_shift_val.csv` | 584 | 483 | 12 | **EXISTS (PROVEN)** |
| **Label Shift Test** | `data/splits/final12_label_shift_test.csv` | 935 | 648 | 12 | **EXISTS (PROVEN)** |
| **Missing Modality** | `data/splits/final12_missing_modality.csv` | 828 | 642 | 12 | **EXISTS (PROVEN)** |

---

## 2. Taxonomy & Imbalance Forensic Audit (Phase B)

| Class Index | Genre Name | Total N | Relative % | Unique Artists | Test Samples (IID) | Imbalance Ratio vs. Min | Status |
|---|---|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | 455 | **32.59x** | Dominant Class |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | 121 | **8.68x** | Secondary Class |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | 43 | **3.09x** | Sufficient |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | 33 | **2.38x** | Sufficient |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | 30 | **2.15x** | Sufficient |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | 29 | **2.08x** | Sufficient |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | 26 | **1.83x** | Sufficient |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | 22 | **1.56x** | Sufficient |
| 8 | `ROCK` | 137 | 2.48% | 20 | 21 | **1.47x** | Sufficient |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | 20 | **1.42x** | Sufficient |
| 10 | `OTHER` | 99 | 1.80% | 54 | 15 | **1.06x** | Verified Semantic Evidence |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | 14 | **1.00x** (Ref) | Small Class |

- **Taxonomy Uniformity:** Exactly 12 classes. No whitespace, casing, or typo deviations.
- **Concentration Metrics:** Gini Index = **$0.6102$**, Entropy = **$2.3885$ bits**.

---

## 3. Train / Val / Test Class Distribution Audit (Phase C)

### A. IID Split ($N=3,860 / 827 / 828$)
- **Status:** **PASS**. All 12 classes are populated across Train, Val, and Test. All Test classes have $\ge 14$ samples.

### B. Artist-Disjoint Split ($N=3,860 / 827 / 828$)
- **Status:** **PASS WITH WARNING**.
  - All 12 classes exist across Train, Val, and Test.
  - `RB_SOUL` has 9 samples in Test (Warning: $< 10$).
  - `OTHER` has 12 samples, `CHILDREN` has 28 samples.

### C. Temporal Shift Split ($N=526 / 54 / 190$ - Verified Years 1967–2026)
- **Status:** 🔴 **CRITICAL ANOMALIES IDENTIFIED**:
  1. `OTHER` has **Train = 0**, Val = 0, Test = 2.
  2. `NHAC_TRINH` has **Test = 0**, Train = 95, Val = 1.
  3. `CHILDREN` has **Val = 0, Test = 0**, Train = 12.
  4. `BOLERO_TRUTINH` has **Val = 0**, Train = 1, Test = 7.
  5. `INSTRUMENTAL` has **Test = 4 (< 5)**.
  - **Scientific Consequence:** In temporal evaluation, Test set only contains **10 of 12 classes**. `CHILDREN` and `NHAC_TRINH` cannot be evaluated at test time, while `OTHER` is evaluated zero-shot without training instances.

### D. Label Shift Split ($N=3,996 / 584 / 935$)
- **Status:** **PASS**. Controlled shift reduces `POP_BALLAD` by $-20.23\%$ in Test while boosting minority classes. All 12 classes have $\ge 27$ test samples.

---

## 4. Artist Distribution & Leakage Audit (Phase D)

- **IID Split:** Train: 2,137, Val: 631, Test: 642 artists. Overlap occurs as expected under uniform random sampling.
- **Artist-Disjoint Split:** Train: 1,908, Val: 428, Test: 411 artists.
  $$\text{Train Artists} \cap \text{Val Artists} = \emptyset \quad (0 \text{ overlap})$$
  $$\text{Train Artists} \cap \text{Test Artists} = \emptyset \quad (0 \text{ overlap})$$
  $$\text{Val Artists} \cap \text{Test Artists} = \emptyset \quad (0 \text{ overlap})$$
  - **Verdict:** **Strict 0% Artist Leakage Confirmed**.

---

## 5. Duplicate & Modality Leakage Audit (Phase E, F, G)

- **`song_id` Duplicates:** $0$ duplicates across splits.
- **`title + artist` Duplicates:** $0$ duplicates across splits.
- **Lyrics Exact Hash Overlap:** $0$ duplicate lyrics between Train and Test.
- **Cover Image Hash Overlap:** 2 hashes shared across splits.
  - *Forensic Investigation:* `RMVM_S_ff90107377` and `RMVM_S_f39ab02c30` share an identical $60,657$-byte image; `RMVM_S_f9d7cffec7` and `RMVM_S_eff62cd2db` share an identical $5,638$-byte image.
  - *Cause:* Upstream streaming platform served a default fallback placeholder image for songs lacking specific album art.

---

## 6. Physical Modality Coverage by Class (Phase H)

| Class | Total N | Physical Lyrics (%) | Physical Covers (%) | Physical Audio (%) | Modality Risk Profile |
|---|---|---|---|---|---|
| `POP_BALLAD` | 3,031 | 2,726 (**89.9%**) | 587 (19.4%) | 0 (0.0%) | Dominant text |
| `BOLERO_TRUTINH` | 807 | 694 (**86.0%**) | 167 (20.7%) | 0 (0.0%) | Strong text |
| `INSTRUMENTAL` | 287 | 217 (**75.6%**) | 44 (15.3%) | 0 (0.0%) | Moderate text |
| `RAP_HIPHOP` | 221 | 111 (**50.2%**) | 21 (9.5%) | 0 (0.0%) | Moderate text |
| `FOLK_TRADITIONAL` | 200 | 82 (**41.0%**) | 18 (9.0%) | 0 (0.0%) | Moderate text |
| `DANCE_EDM` | 193 | 149 (**77.2%**) | 21 (10.9%) | 0 (0.0%) | Strong text |
| `REVOLUTIONARY` | 170 | 23 (**13.5%**) | 4 (2.4%) | 0 (0.0%) | Sparse text & visual |
| `NHAC_TRINH` | 145 | 12 (**8.3%**) | 2 (1.4%) | 0 (0.0%) | Highly sparse |
| `ROCK` | 137 | 15 (**10.9%**) | 6 (4.4%) | 0 (0.0%) | Highly sparse |
| `RB_SOUL` | 132 | 14 (**10.6%**) | 4 (3.0%) | 0 (0.0%) | Highly sparse |
| `OTHER` | 99 | 0 (**0.0%**) | 14 (14.1%) | 0 (0.0%) | **Zero physical lyrics** |
| `CHILDREN` | 93 | 74 (**79.6%**) | 14 (15.1%) | 0 (0.0%) | Strong text |

---

## 7. Model Preprocessing Isolation (Phase L)

- **TF-IDF Fitting:** Vectorizer is fitted strictly on `final12_iid_train.csv` ($2,877$ texts). No test text touches the vocabulary.
- **Class Weights:** Computed strictly on Train split labels.
- **Verdict:** **PASSED (Zero preprocessing leakage)**.
