# RM-VMusic: Full Project Reconnaissance & Research State Report
**Phase 7A — Comprehensive Audit & Single Source of Truth**  
**Audit Date:** 2026-08-28  
**Project:** RM-VMusic (*Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift*)  
**Execution Environment:** Windows | Shell: PowerShell | Python: 3.12 (UTF-8)

---

## 1. Executive Summary

This report delivers a definitive, empirical, filesystem-verified audit of the entire **RM-VMusic** research codebase. Every number, path, and finding in this report has been directly extracted and validated from local disk files, active CSV catalogs, PyTorch checkpoint artifacts, and evaluation metric logs.

### Key High-Level Findings
1. **Catalog & Metadata Ground Truth:**
   - The master catalog contains **8,738** total records across 3 tiers (Tier A: 4,157; Tier B: 1,259; Tier C: 3,322).
   - The verified trainable dataset contains **5,416** records (Tier A + Tier B) across **11 core genre classes**.
   - Zero duplicated records exist across `song_id`, `(title, artist)`, or `source_id`.
2. **Physical Asset Discrepancy (CRITICAL SCIENTIFIC FACT):**
   - **Physical Audio (`data/audio/`):** **0 files (0.00% physical coverage)**. No local audio waveforms (`.mp3`, `.wav`, `.flac`) exist on disk.
   - **Physical Cover Art (`data/covers/`):** **412 files (7.61% physical coverage)**. All 412 are valid `.jpg` files.
   - **Physical Lyrics (`data/lyrics/`):** **4,117 files (76.02% physical coverage)**. All 4,117 are valid `.txt` files.
   - **Baseline & Phase 5 Feature Extraction Reality:** In existing experimental runs (`train_baseline.py`, `train_proposed.py`), audio and cover representations were derived from deterministic SHA-256 pseudo-hash embeddings due to missing physical audio and low cover availability. Only lyrics used real textual feature extraction (TF-IDF).
3. **Taxonomy & "OTHER" Class Status:**
   - **Active/Evaluated Taxonomy:** **11 Classes** (`POP_BALLAD`, `BOLERO_TRUTINH`, `INSTRUMENTAL`, `RAP_HIPHOP`, `FOLK_TRADITIONAL`, `DANCE_EDM`, `REVOLUTIONARY`, `NHAC_TRINH`, `ROCK`, `RB_SOUL`, `CHILDREN`).
   - **OTHER Class Status:** `NOT IMPLEMENTED` in active training configs/checkpoints. An experimental script (`build_final_12class_dataset.py`) produced an unverified 12-class candidate file (`final_12class_metadata.csv`, N=5,514 with 98 OTHER samples), but **no baseline or proposed models have been trained or verified with physical audio on 12 classes**.
4. **Distribution Shift Benchmarks:**
   - All 5 benchmark splits (**IID**, **Artist Disjoint**, **Temporal**, **Label Shift**, **Missing Modality**) are generated and mathematically verified.
   - Strict 0% artist leakage is achieved on `final_artist_disjoint` (Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0).
   - Temporal split is severely constrained: only **768 / 5,416 (14.18%)** samples possess verified release years; 4,648 samples are excluded.

---

## 2. Project Filesystem Tree & Directory Analysis

```text
RM-VMusic/
├── .gitignore                      [3,045 bytes, comprehensive ML/Python ignore rules]
├── README.md                       [4,315 bytes, core overview & master schema definition]
├── requirements.txt                [92 bytes, pandas, numpy, requests, tqdm, scikit-learn, scipy]
├── configs/                        [2 YAML configuration files]
│   ├── baseline.yaml               [1,322 bytes, 11-class baseline configuration]
│   └── proposed.yaml               [2,560 bytes, UAD-Fusion ablation ladder configuration]
├── data/
│   ├── raw/                        [5 files, 147.80 MB]
│   │   ├── sunbv56_eval.jsonl      [643 records, 227 KB]
│   │   ├── sunbv56_pilot_train.jsonl [1,500 records, 29.46 MB]
│   │   ├── sunbv56_train_full.jsonl  [7,201 records, 116.23 MB]
│   │   ├── vietlyrics_train_7k.csv [7,433 records, 1.67 MB]
│   │   └── vietlyrics_val_1k.csv   [995 records, 230 KB]
│   ├── audio/                      [0 files, 0 MB — DIRECTORY EXISTS BUT IS EMPTY]
│   ├── covers/                     [412 files, 17.65 MB, all .jpg]
│   ├── lyrics/                     [4,117 files, 8.23 MB, all .txt]
│   ├── processed/                  [10 CSV files, 35.59 MB]
│   │   ├── master_metadata.csv     [8,738 rows, master 22-column catalog]
│   │   ├── trainable_metadata.csv  [5,416 rows, Tier A + Tier B]
│   │   ├── final_trainable_metadata.csv [5,416 rows, canonical 11-class dataset]
│   │   ├── trainable_physical_verified.csv [5,416 rows, modality status tracking]
│   │   ├── final_12class_metadata.csv [5,514 rows, experimental 12-class file]
│   │   ├── manual_annotation_queue.csv [3,322 rows, Tier C annotation candidates]
│   │   ├── rejected_records.csv    [321 rows, discarded malformed/duplicate tracks]
│   │   ├── recovery_blocked.csv    [5,876 rows, failed download logs]
│   │   ├── audio_collection_queue.csv [5,416 rows, audio download targets]
│   │   └── cover_collection_queue.csv [5,416 rows, cover download targets]
│   └── splits/                     [31 CSV split files, 65.46 MB]
├── docs/                           [6 Markdown documentation files, 28.5 KB]
├── outputs/                        [15 files, 74.38 MB]
│   ├── checkpoints/                [12 PyTorch .pt model weights, 74.23 MB]
│   │   ├── baseline_*.pt           [7 baseline models: audio, lyrics, cover & combinations]
│   │   └── proposed/               [5 ablation models: Model A through Model E]
│   └── metrics/                    [3 JSON summary logs, 153 KB]
│       ├── all_baselines_summary.json
│       ├── phase6_stress_stats_summary.json
│       └── proposed/proposed_results_summary.json
├── reports/                        [78 files: 28 .md, 25 .csv, 25 .png figures]
├── scripts/                        [41 files: 36 .py scripts, 5 .pyc compiled files]
├── src/                            [DOES NOT EXIST]
└── tests/                          [DOES NOT EXIST]
```

### Directory Audit Summary Table

| Directory | Exists? | File Count | File Types | Total Size | Empty Files | Corrupt Files | Notes |
|---|---|---|---|---|---|---|---|
| `data/raw/` | **YES** | 5 | `.jsonl` (3), `.csv` (2) | 147.80 MB | 0 | 0 | Raw cached datasets |
| `data/audio/` | **YES** | **0** | None | **0 MB** | 0 | 0 | **Empty directory (0 physical audio)** |
| `data/covers/` | **YES** | 412 | `.jpg` (412) | 17.65 MB | 0 | 0 | Valid JPEG images |
| `data/lyrics/` | **YES** | 4,117 | `.txt` (4,117) | 8.23 MB | 0 | 0 | Valid UTF-8 text files |
| `data/processed/`| **YES** | 10 | `.csv` (10) | 35.59 MB | 0 | 0 | Master, trainable, queues |
| `data/splits/` | **YES** | 31 | `.csv` (31) | 65.46 MB | 0 | 0 | 5 shift splits across versions |
| `configs/` | **YES** | 2 | `.yaml` (2) | 3.88 KB | 0 | 0 | Baseline and UAD-Fusion configs |
| `scripts/` | **YES** | 41 | `.py` (36), `.pyc` (5) | 507 KB | 0 | 0 | Data processing & training scripts |
| `src/` | **NO** | 0 | N/A | 0 MB | N/A | N/A | Modular source package not built |
| `tests/` | **NO** | 0 | N/A | 0 MB | N/A | N/A | Unit tests not formalized |
| `reports/` | **YES** | 78 | `.md` (28), `.csv` (25), `.png` (25) | 4.40 MB | 0 | 0 | Scientific reports & plots |
| `docs/` | **YES** | 6 | `.md` (6) | 28.5 KB | 0 | 0 | Dataset cards & specifications |
| `outputs/` | **YES** | 15 | `.pt` (12), `.json` (3) | 74.38 MB | 0 | 0 | Checkpoints & metric summaries |

---

## 3. Physical Asset Audit

Direct inspection of filesystem files versus metadata columns establishes the exact physical status of the multimodal dataset.

### Physical Modality Coverage Table

| Modality | Metadata Claims Available | Streaming/Remote URL Available | Physical Files on Disk | Valid Physical Files | Physical Coverage (N=5,416) |
|---|---|---|---|---|---|
| **Audio** | 5,416 (`audio_url`) | 5,416 | **0** | **0** | **0.00%** |
| **Covers** | 413 | 413 | **412** | **412** | **7.61%** |
| **Lyrics** | 4,117 | 4,117 | **4,117** | **4,117** | **76.02%** |

### Physical Modality Combinations on Disk (N=5,416 Trainable Tracks)

```text
├── Full Multimodal (Audio + Lyrics + Cover on disk): 0 tracks  (0.00%)
├── Dual Modality:
│   ├── Lyrics + Cover on disk:                      99 tracks  (1.83%)
│   ├── Audio + Lyrics on disk:                       0 tracks  (0.00%)
│   └── Audio + Cover on disk:                        0 tracks  (0.00%)
├── Single Modality:
│   ├── Lyrics Only on disk:                      4,018 tracks (74.19%)
│   ├── Cover Only on disk:                         313 tracks  (5.78%)
│   └── Audio Only on disk:                           0 tracks  (0.00%)
└── Zero Physical Assets on disk:                   986 tracks (18.21%)
```

> [!CAUTION]
> **CRITICAL SCIENTIFIC DISCLOSURE:**
> There is **zero physical audio** in `data/audio/`. Any evaluation reporting "Audio-only" or "Audio+Lyrics+Cover" in Phase 4/5 baseline runs was computed using deterministic SHA-256 hash pseudo-features (`scripts/train_baseline.py` lines 90–108), not physical waveforms.

---

## 4. Metadata Schema & Source of Truth Audit

### File Classification & Hierarchy

| File Path | Rows | Columns | Role / Status | Recommendation |
|---|---|---|---|---|
| `data/processed/master_metadata.csv` | **8,738** | 22 | **Master Catalog (Source of Truth)** across Tier A, B, C | Keep as master index |
| `data/processed/final_trainable_metadata.csv` | **5,416** | 22 | **Canonical 11-Class Trainable Dataset** (Tier A + B) | Primary trainable reference |
| `data/processed/trainable_metadata.csv` | 5,416 | 22 | Intermediate 11-class trainable file | Retain for provenance |
| `data/processed/trainable_physical_verified.csv` | 5,416 | 13 | Asset status verification report table | Retain |
| `data/processed/final_12class_metadata.csv` | **5,514** | 18 | Experimental 12-class candidate file (+98 OTHER) | Unvalidated proposal |
| `data/processed/manual_annotation_queue.csv` | 3,322 | 9 | Tier C songs needing manual review | Retain for human-in-the-loop |
| `data/processed/rejected_records.csv` | 321 | 23 | Tracked rejected/malformed raw records | Retain for auditability |
| `data/processed/recovery_blocked.csv` | 5,876 | 5 | Logs of failed download attempts | Retain |
| `data/processed/audio_collection_queue.csv` | 5,416 | 9 | Queue for audio collection | Pending execution |
| `data/processed/cover_collection_queue.csv` | 5,416 | 10 | Queue for cover collection | Pending execution |

### Canonical Master Metadata Schema (22 Fields)

1. `song_id` *(string)*: Unique canonical identifier (e.g. `RMVM_S_0007886478`)
2. `title` *(string)*: Standardized song title (Unicode NFC)
3. `artist` *(string)*: Standardized artist name(s)
4. `artist_id` *(string)*: Unique deterministic artist hash ID (`ART_<hash>`)
5. `album` *(string)*: Album name or empty
6. `album_id` *(string)*: Album ID (`ALB_<hash>`) or empty
7. `source_genre` *(string)*: Raw genre string from source
8. `genre` *(string)*: Standardized 11-class genre category
9. `label_source` *(string)*: Provenance method (`source_exact`, `expert_verified`, `source_cross_verified`)
10. `tier` *(string)*: Quality tier (`TIER_A`, `TIER_B`, `TIER_C`)
11. `release_year` *(float/int)*: Release year (present in 802 master tracks, 768 trainable tracks)
12. `release_year_source` *(string)*: Origin of release year tag
13. `audio_path` *(string)*: Expected local audio path (`data/audio/<song_id>.mp3`)
14. `audio_url` *(string)*: Streaming / remote reference URL
15. `lyrics` *(string)*: Raw lyric text string
16. `cover_path` *(string)*: Expected local cover path (`data/covers/<song_id>.jpg`)
17. `cover_url` *(string)*: Remote cover image URL
18. `source` *(string)*: Origin dataset identifier (`sunbv56`, `vietlyrics`)
19. `source_id` *(string)*: Source dataset identifier
20. `annotation_status` *(string)*: `cross_verified`, `normalized`, or `needs_annotation`
21. `annotator_id` *(string)*: Machine or human annotator ID (`pipeline_v1`, `source_curated`)
22. `annotation_agreement` *(float)*: Agreement confidence score (1.00 for Tier A, 0.80–0.90 for Tier B)

---

## 5. Dataset Count Reconciliation

Direct verification against CSV files shows exact consistency across all primary tables.

| Metric | Reported in Docs | Actual Filesystem Count | Discrepancy | Explanatory Source File |
|---|---|---|---|---|
| **Master Records** | 8,738 | **8,738** | 0 | `data/processed/master_metadata.csv` |
| **Trainable Records (11-Class)** | 5,416 | **5,416** | 0 | `data/processed/final_trainable_metadata.csv` |
| **Tier A Samples** | 4,157 | **4,157** | 0 | High confidence cross-verified |
| **Tier B Samples** | 1,259 | **1,259** | 0 | Normalized source genre |
| **Tier C Samples** | 3,322 | **3,322** | 0 | In `manual_annotation_queue.csv` |
| **Rejected Records** | 321 | **321** | 0 | `data/processed/rejected_records.csv` |
| **Unique Song IDs** | 5,416 | **5,416** | 0 | 0 duplicate song IDs in trainable set |
| **Unique Artists** | 2,712 | **2,712** | 0 | Trainable set has 2,712 unique artists |
| **Duplicate (Title + Artist)** | 0 | **0** | 0 | Deduplication is clean |
| **Duplicate Source IDs** | 0 | **0** | 0 | Clean 1-to-1 source indexing |
| **Duplicate Audio URLs** | 0 | **0** | 0 | Clean remote endpoint index |
| **Verified Release Years** | 768 | **768** | 0 | 14.18% coverage (1967–2026) |

---

## 6. Taxonomy Audit & "OTHER" Class Evaluation

### Active Evaluated 11-Class Taxonomy Breakdown (`final_trainable_metadata.csv`, N=5,416)

| # | Genre Class | Sample Count | Percentage | Unique Artists | Imbalance Ratio (vs Min) |
|---|---|---|---|---|---|
| 1 | **POP_BALLAD** | 3,031 | 55.96% | 1,890 | 32.59x |
| 2 | **BOLERO_TRUTINH** | 807 | 14.90% | 501 | 8.68x |
| 3 | **INSTRUMENTAL** | 287 | 5.30% | 141 | 3.09x |
| 4 | **RAP_HIPHOP** | 221 | 4.08% | 111 | 2.38x |
| 5 | **FOLK_TRADITIONAL** | 200 | 3.69% | 77 | 2.15x |
| 6 | **DANCE_EDM** | 193 | 3.56% | 139 | 2.08x |
| 7 | **REVOLUTIONARY** | 170 | 3.14% | 31 | 1.83x |
| 8 | **NHAC_TRINH** | 145 | 2.68% | 23 | 1.56x |
| 9 | **ROCK** | 137 | 2.53% | 20 | 1.47x |
| 10 | **RB_SOUL** | 132 | 2.44% | 27 | 1.42x |
| 11 | **CHILDREN** | 93 | 1.72% | 41 | 1.00x |
| **Total** | **11 Classes** | **5,416** | **100.00%** | **2,712** | **Max Imbalance: 32.59x** |

### Status of the "OTHER" Class

```text
OTHER CLASS STATUS = NOT IMPLEMENTED
```
- **In Active Baseline & Proposed Models:** The class `OTHER` **does not exist** (`num_classes: 11` in `configs/baseline.yaml` and `configs/proposed.yaml`).
- **In Experimental Candidate File:** `data/processed/final_12class_metadata.csv` (N=5,514) contains 98 tracks annotated as `OTHER` (1.78% of dataset, 53 unique artists), but this dataset version has **never been evaluated with actual physical audio or formally benchmarked**.

---

## 7. Train / Val / Test & Data Isolation Audit

### Split Breakdown & Overlap Analysis (Canonical 11-Class Benchmark)

| Split Name | Train Size | Val Size | Test Size | Total | Song Leakage (Tr∩Val / Tr∩Te / Val∩Te) | Artist Leakage (Tr∩Val / Tr∩Te / Val∩Te) |
|---|---|---|---|---|---|---|
| **IID** (`final_iid_*.csv`) | 3,791 (70%) | 812 (15%) | 813 (15%) | 5,416 | **0 / 0 / 0** | 323 / 336 / 181 (Expected IID overlap) |
| **Artist Disjoint** (`final_artist_disjoint_*.csv`) | 3,791 (70%) | 812 (15%) | 813 (15%) | 5,416 | **0 / 0 / 0** | **0 / 0 / 0 (STRICT 0% LEAKAGE)** |
| **Temporal Shift** (`final_temporal_*.csv`) | 526 (68.5%) | 54 (7.0%) | 188 (24.5%) | 768 | **0 / 0 / 0** | 15 / 22 / 18 |
| **Label Shift** (`final_label_shift_*.csv`) | 3,855 (71.2%)| 775 (14.3%)| 786 (14.5%)| 5,416 | **0 / 0 / 0** | 271 / 298 / 155 |
| **Missing Modality** (`final_missing_modality.csv`) | — | — | 5,416 | 5,416 | — | Stress testing under synthetic modality dropout |

### Artist Disjoint Strictness Proof
- **Train Partition Artists:** 1,837 unique artists
- **Val Partition Artists:** 474 unique artists
- **Test Partition Artists:** 401 unique artists
- **Exact Set Intersections:**
  $$\text{Train} \cap \text{Val} = \emptyset \quad (0)$$
  $$\text{Train} \cap \text{Test} = \emptyset \quad (0)$$
  $$\text{Val} \cap \text{Test} = \emptyset \quad (0)$$

---

## 8. Distribution Shift Benchmark Audit

| Shift Benchmark | Shift Type / Objective | Construction Condition | Train Partition | Val Partition | Test Partition | Total Samples | Script & Config |
|---|---|---|---|---|---|---|---|
| **1. IID** | In-Distribution Control | Stratified 70/15/15 by genre | 3,791 | 812 | 813 | 5,416 | `create_final_splits.py`, `baseline.yaml` |
| **2. Artist Disjoint** | Generalization to Unseen Artists | Group-stratified by `artist_id` | 3,791 | 812 | 813 | 5,416 | `create_final_splits.py`, `baseline.yaml` |
| **3. Temporal Shift** | Generalization Across Eras | Time-sorted on verified `release_year` | 526 (1967–2018) | 54 (2019–2020) | 188 (2021–2026) | 768 | `create_final_splits.py`, `baseline.yaml` |
| **4. Label Shift** | Prior Class Distribution Change | Controlled sampling inducing prior shift | 3,855 | 775 | 786 | 5,416 | `create_final_splits.py`, `baseline.yaml` |
| **5. Missing Modality** | Sensor Failure / Incomplete Data | Modality masking ($p=0.20, 0.50, 1.00$) | — | — | 5,416 | 5,416 | `create_final_splits.py`, `proposed.yaml` |

---

## 9. Baseline Experiments & Proposed Method Audit

### Baseline Setup (`configs/baseline.yaml`, `scripts/train_baseline.py`)
- **Loss:** Class-Balanced Cross-Entropy ($w_c = \frac{N}{C \cdot N_c}$ computed on train split only).
- **Optimizer:** Adam ($lr = 0.001$, weight decay = $10^{-4}$).
- **Early Stopping:** Patience 8 epochs (max 35 epochs), batch size 64, seed 42.
- **Architectures:** Modality-specific encoders projected to common hidden dimension (256-d) with concatenation.

### Empirical Baseline Results (`outputs/metrics/all_baselines_summary.json`)

#### A. Modality Ablation (Evaluated on IID Test Set, N=813)

| Modality Combination | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---|---|---|---|
| `audio_only`* | 0.0815 | 0.0575 | 0.1005 | 0.0815 |
| `cover_only`* | 0.1210 | 0.0410 | 0.1437 | 0.1026 |
| `lyrics_only` (TF-IDF) | **0.3840** | **0.2364** | **0.4457** | **0.2874** |
| `audio_cover`* | 0.1630 | 0.0859 | 0.2085 | 0.1059 |
| `lyrics_cover` | 0.4383 | 0.2544 | 0.4884 | 0.3071 |
| `audio_lyrics` | **0.5395** | **0.2433** | **0.5539** | **0.2575** |
| `audio_lyrics_cover` (Full Concat) | 0.4914 | 0.2584 | 0.5326 | 0.2811 |

*\*Note: Audio and cover feature extractors used deterministic pseudo-hash vectors in Phase 4/5.*

#### B. Baseline Distribution Shift Degradation (Full Concat Model)

| Benchmark Split | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Macro-F1 Drop vs IID |
|---|---|---|---|---|---|
| **IID** | 0.4914 | 0.2584 | 0.5326 | 0.2811 | — |
| **Artist Disjoint** | 0.5301 | 0.2459 | 0.5428 | 0.2863 | **-4.84%** |
| **Label Shift** | 0.3658 | 0.2524 | 0.3550 | 0.2747 | **-2.32%** |
| **Missing Modality** | 0.5211 | 0.1663 | 0.4941 | 0.1750 | **-35.64%** |
| **Temporal Shift** | 0.2128 | 0.1573 | 0.2333 | 0.2313 | **-39.13%** |

---

### Proposed Method: UAD-Fusion (`scripts/train_proposed.py`, `outputs/metrics/proposed/`)

#### Ablation Ladder (IID Test Split)

| Model Variation | Key Components | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| **Model A** | Baseline Concat | 0.4914 | 0.2584 | 0.5326 | 0.2811 |
| **Model B** | + Dynamic Uncertainty/Reliability | **0.5284** | 0.2576 | **0.5534** | 0.2775 |
| **Model C** | + Controlled Modality Dropout ($p=0.20$) | 0.4728 | 0.2613 | 0.5170 | 0.2697 |
| **Model D** | + Distribution Invariance Regularizer | 0.4728 | **0.2630** | 0.5152 | 0.2697 |
| **Model E** | Full UAD-Fusion (+ SupCon Loss) | 0.4704 | 0.2543 | 0.5147 | 0.2622 |

#### Multi-Seed Stability (Model E over Seeds 42, 123, 2026)
- **IID Macro-F1:** $0.2554 \pm 0.0003$
- **Artist Disjoint Macro-F1:** $0.2232 \pm 0.0137$
- **Missing Modality Macro-F1:** $0.1693 \pm 0.0074$

---

## 10. Code & Pipeline Inventory

| Script Name | Purpose | Primary Inputs | Primary Outputs | Status | Used in Final Pipeline? |
|---|---|---|---|---|---|
| `build_dataset.py` | Initial dataset curation & tiering | `data/raw/*.jsonl`, `.csv` | `data/processed/master_metadata.csv` | Legacy/Historical | Yes (Phase 1-3) |
| `audit_sources.py` | Audits remote Vietnamese music endpoints | Web requests | `docs/dataset_audit.md` | Audit tool | Yes |
| `check_duplicates.py` | Audits exact & normalized duplicates | Processed metadata CSV | Console output | Utility | Yes |
| `check_artist_leakage.py` | Formally checks artist disjointness | `data/splits/*.csv` | Console verification | Verification | Yes |
| `create_splits.py` | Generates first version of 5 shift splits | `trainable_metadata.csv` | `data/splits/*.csv` | Legacy | Superceded |
| `create_final_splits.py` | Generates final canonical 11-class splits | `final_trainable_metadata.csv` | `data/splits/final_*.csv` | **Production** | **YES (Canonical)** |
| `build_final_dataset.py` | Builds final 11-class dataset & card | `master_metadata.csv` | `final_trainable_metadata.csv` | **Production** | **YES (Canonical)** |
| `train_baseline.py` | Trains 7 baseline modality models | `data/splits/*.csv` | `outputs/checkpoints/baseline_*.pt` | **Production** | **YES (Phase 4)** |
| `run_all_baselines.py` | Master execution runner for baselines | `configs/baseline.yaml` | `outputs/metrics/all_baselines_summary.json` | **Production** | **YES (Phase 4)** |
| `train_proposed.py` | Trains UAD-Fusion ablation ladder | `configs/proposed.yaml` | `outputs/checkpoints/proposed/*.pt` | **Production** | **YES (Phase 5)** |
| `run_proposed_experiments.py`| Master execution runner for UAD-Fusion | `configs/proposed.yaml` | `outputs/metrics/proposed/*.json` | **Production** | **YES (Phase 5)** |
| `evaluate.py` | Evaluation & confusion matrix engine | Model checkpoints | Metric dictionaries | **Production** | **YES** |
| `evaluate_proposed.py` | Diagnostic evaluator for UAD-Fusion | Proposed checkpoints | Reports & figures | **Production** | **YES** |
| `phase6_audit.py` | Audit of leakage safeguards & data isolation| Metadata & splits | `reports/phase5_audit.md` | Verification | Yes |
| `phase6_stress_and_stats.py`| Multi-seed, bootstrap CIs, calibration | Proposed checkpoints | `outputs/metrics/phase6_*.json` | **Production** | **YES (Phase 6)** |
| `phase6_plot_figures.py` | Generates publication figures | Metric summary JSONs | `reports/figures/*.png` | Visualization | Yes |
| `recover_lyrics.py` | Materializes metadata lyrics to `.txt` files | `master_metadata.csv` | `data/lyrics/*.txt` | **Production** | **YES (Phase 6)** |
| `recover_covers.py` | Downloads & validates physical `.jpg` covers | `master_metadata.csv` | `data/covers/*.jpg` | **Production** | **YES (Phase 6)** |
| `build_final_12class_dataset.py` | Generates 12-class dataset (+98 OTHER) | `master_metadata.csv` | `final_12class_metadata.csv` | Experimental | Candidate |
| `create_final_12class_splits.py` | Generates 12-class benchmark splits | `final_12class_metadata.csv` | `data/splits/final_12class_*.csv` | Experimental | Candidate |
| `collect_physical_audio.py` | Audio queue & download worker | Remote endpoints | `data/audio/*.mp3` | **Prepared / Pending Execution** | Incomplete |
| `collect_physical_covers.py` | Cover queue & image download worker | Remote endpoints | `data/covers/*.jpg` | **Prepared / Pending Execution** | Incomplete |

---

## 11. Configuration Inventory

| Config File | Defined Classes | Dataset Paths | Modalities Defined | Loss & Optimizers | Special Parameters |
|---|---|---|---|---|---|
| `configs/baseline.yaml` | **11 Classes** (No OTHER) | `master_metadata.csv`, `trainable_metadata.csv` | `audio_only`, `lyrics_only`, `cover_only`, `audio_lyrics`, `audio_cover`, `lyrics_cover`, `audio_lyrics_cover` | Balanced Cross-Entropy, Adam ($lr=10^{-3}$) | `early_stopping_patience: 8`, `dropout: 0.3` |
| `configs/proposed.yaml` | **11 Classes** (No OTHER) | `master_metadata.csv`, `trainable_metadata.csv` | Multimodal joint (Audio, Lyrics, Cover) | Weighted CE + SupCon + Uncertainty Loss + Invariance Loss | Modality Dropout ($p=0.20$), Seeds: `[42, 123, 2026]`, Ablation Models A–E |

---

## 12. Documentation & Reports Inventory

### Documentation Classification (`docs/`)
- `docs/genre_taxonomy.md` (**Authoritative**): Canonical genre mapping rules, definitions, and tier hierarchy.
- `docs/final_dataset_card.md` (**Authoritative**): Final 11-class dataset card and provenance report.
- `docs/final_dataset_card_v2.md` (**Candidate / Unvalidated**): Proposed 12-class dataset card.
- `docs/dataset_audit.md` (**Historical / Authoritative**): Technical audit of the 4 raw data sources.
- `docs/targeted_collection_guide.md` (**Authoritative**): Acquisition guidelines for rare classes.
- `docs/dataset_versioning.md` (**Authoritative**): Version numbering and lifecycle specification.

### Key Reports Classification (`reports/`)
- `reports/baseline_results.md` (**Authoritative for Phase 4**): Baseline results across 7 modalities and 5 shifts.
- `reports/proposed_method.md` (**Authoritative for Phase 5**): UAD-Fusion formulation and empirical gains.
- `reports/phase6_scientific_conclusion.md` (**Authoritative for Phase 6**): Formal separation of facts, inferences, and hypotheses.
- `reports/phase6_data_completion_report.md` (**Authoritative for Phase 6**): First formal identification of physical asset gap.
- `reports/physical_modality_report.md` (**Authoritative for Phase 6**): Exact physical file matrix on disk.
- `reports/physical_asset_audit.md` (**Authoritative for Phase 7**): Song matching verification.

---

## 13. Research State Synthesis

### Research Problem
Vietnamese music genre classification suffers from severe modality unreliability, heavy class imbalance, and significant performance collapse under real-world distribution shifts:
1. **Artist Shift:** Models memorize specific artist stylistic signatures rather than genre-level features.
2. **Temporal Shift:** Musical arrangement conventions evolve over decades (pre-2018 vs post-2021).
3. **Missing Modality / Sensor Failure:** Audio, lyrics, or album art may be unavailable or corrupted during streaming inference.

### Scientific Facts Established by Evidence
1. **Modality Dominance:** Lyrics (TF-IDF) is currently the strongest valid modality ($F_1 = 0.2364$), far outperforming uninformative pseudo-features.
2. **Severe Temporal Collapse:** Classification Macro-F1 collapses by **-39.13%** under temporal shift (1967–2018 train $\to$ 2021–2026 test).
3. **Severe Missing Modality Collapse:** Classification Macro-F1 collapses by **-35.64%** when modalities are dropped without uncertainty handling.
4. **Dynamic Reliability Defense:** Dynamic uncertainty weighting (UAD-Fusion Model B) improves accuracy to 52.84% (+3.70% over baseline concat).

---

## 14. Data Quality Scorecard

| Dimension | Score / Status | Evidence / Verification |
|---|---|---|
| **Label Quality** | **HIGH (9.2/10)** | 76.75% Tier A (cross-verified across multiple independent catalogs), 23.25% Tier B. Zero duplicate records. |
| **Audio Physical Availability** | **CRITICAL FAILURE (0.0/10)** | **0 physical files on disk (0.00% coverage)**. All audio URLs remain remote. |
| **Lyrics Physical Availability** | **HIGH (7.6/10)** | **4,117 valid physical text files (76.02% coverage)** on disk. |
| **Cover Physical Availability** | **LOW (1.0/10)** | **412 valid physical JPEG files (7.61% coverage)** on disk. |
| **Class Balance** | **POOR (3.5/10)** | Severe imbalance (POP_BALLAD: 55.96%, CHILDREN: 1.72%, ratio: 32.59x). |
| **Artist Diversity** | **EXCELLENT (9.5/10)** | 2,712 unique artists across 5,416 tracks (avg 2.0 tracks/artist). |
| **Deduplication Control** | **PERFECT (10/10)** | 0 duplicates across `song_id`, `(title, artist)`, `source_id`, `audio_url`. |
| **Artist Leakage Prevention**| **PERFECT (10/10)** | Strict 0% artist overlap on `final_artist_disjoint` (Train ∩ Val = 0, Train ∩ Test = 0). |
| **Temporal Coverage** | **LOW (2.5/10)** | Only 768 / 5,416 (14.18%) have verified release years. |
| **Distribution Shift Diversity**| **EXCELLENT (9.0/10)** | 5 well-defined benchmarks (IID, Artist, Temporal, Label Shift, Missing Modality). |
| **Code Reproducibility** | **HIGH (8.5/10)** | Fixed random seeds (42, 123, 2026), deterministic dataloaders, modular PyTorch scripts. |
| **Scientific Defensibility** | **MEDIUM (5.0/10)** | High on methodology, splits, and lyrics; but **untenable for publication until physical audio is downloaded and real acoustic features are extracted**. |

---

## 15. Critical Problems & Risk Hierarchy

### Level 1: CRITICAL (Scientific Invalidation Risk)
1. **Missing Physical Audio Waveforms:** `data/audio/` contains 0 files. Training scripts currently use pseudo-hash features for audio. **No multimodal audio claims can be scientifically defended in a peer-reviewed paper until real acoustic features (e.g. Mel-Spectrograms, Wav2Vec2, or CQT) are computed from real audio files.**
2. **Missing Physical Cover Images:** Only 412 / 5,416 tracks (7.61%) possess physical image files on disk.

### Level 2: HIGH (Methodological & Dataset Quality Risks)
3. **Severe Class Imbalance:** POP_BALLAD accounts for 55.96% of the dataset, while 5 minority classes each account for $<3.5\%$.
4. **Low Temporal Metadata Coverage:** Only 14.18% of samples have verified release years, restricting the temporal split to 768 tracks.
5. **OTHER Class Ambiguity:** `OTHER` is implemented in an unverified 12-class metadata file (98 samples) but is not implemented in active models or verified against physical audio.

### Level 3: MEDIUM (Engineering & Architecture Improvements)
6. **No Modular `src/` Package:** Scripts directly import from local files rather than an installable research package (`pip install -e .`).
7. **No Formal Unit Test Suite:** `tests/` directory does not exist; validation is performed via standalone verification scripts.

---

## 16. Completed vs. Remaining Tasks

### Completed Tasks (DONE)

| Task | Category | Evidence / Artifact |
|---|---|---|
| Master Catalog Curation (N=8,738) | Dataset | `data/processed/master_metadata.csv` |
| Trainable Dataset Construction (N=5,416) | Dataset | `data/processed/final_trainable_metadata.csv` |
| 100% Deduplication & Provenance Indexing | Data Quality | `reports/final_dedup_report.md` |
| Physical Lyrics Materialization (N=4,117) | Data Asset | `data/lyrics/*.txt` (8.23 MB) |
| Physical Cover Art Materialization (N=412) | Data Asset | `data/covers/*.jpg` (17.65 MB) |
| 5 Distribution Shift Split Construction | Benchmarking | `data/splits/final_*.csv` |
| Strict 0% Artist Leakage Verification | Benchmarking | `reports/final_artist_leakage_report.md` |
| Multimodal Baseline Architecture Implementation | Modeling | `scripts/train_baseline.py`, `configs/baseline.yaml` |
| Baseline Benchmark Execution across 7 Modalities | Experiments | `outputs/metrics/all_baselines_summary.json` |
| Proposed UAD-Fusion Architecture Implementation | Modeling | `scripts/train_proposed.py`, `configs/proposed.yaml` |
| UAD-Fusion Ablation Ladder Execution (Models A–E) | Experiments | `outputs/metrics/proposed/proposed_results_summary.json` |
| Multi-Seed Statistical Validation (3 Seeds) | Statistics | `outputs/metrics/phase6_stress_stats_summary.json` |
| Publication Figures Generation (25 Figures) | Visualization | `reports/figures/*.png` |
| Comprehensive Markdown Documentation (53 Reports, 6 Docs)| Documentation | `reports/*.md`, `docs/*.md` |

### Remaining Tasks (NOT DONE)

| Task | Reason Not Done | Priority |
|---|---|---|
| **Materialize Physical Audio Files (`data/audio/`)** | Requires downloading audio waveforms for the 5,416 tracks | **P0 (CRITICAL)** |
| **Materialize Remaining Physical Cover Art** | Need to download remaining 5,004 covers from verified URLs | **P1 (HIGH)** |
| **Extract Real Acoustic & Visual Features** | Blocked until physical audio and covers exist on disk | **P0 (CRITICAL)** |
| **Formal Decision on Taxonomy (11 vs 12 Class / OTHER)** | Requires human consensus on whether to include OTHER | **P1 (HIGH)** |
| **Re-train Baselines with Real Acoustic & Visual Features**| Blocked until real feature matrices are extracted | **P0 (CRITICAL)** |
| **Re-train Proposed UAD-Fusion with Real Features** | Blocked until real baselines are trained | **P0 (CRITICAL)** |
| **Enrich Release Years for Temporal Split** | Requires scraping/querying release metadata for the 4,648 missing tracks | **P2 (MEDIUM)** |
| **Refactor into Modular Package (`src/rm_vmusic/`)** | Engineering cleanup | **P3 (LOW)** |

---

## 17. Recommended Next Research Pipeline

```text
PHASE 7B: Physical Asset Materialization
├── Step 1: Download physical audio waveforms into data/audio/ (target: >=4,000 tracks)
└── Step 2: Download remaining physical cover images into data/covers/
        │
        ▼
PHASE 7C: Real Multimodal Feature Extraction
├── Step 3: Extract acoustic representations (e.g. Mel-Spectrograms, MFCCs, or OpenL3/Wav2Vec2)
├── Step 4: Extract visual representations (e.g. ResNet-50 / ViT image embeddings)
└── Step 5: Extract linguistic representations (PhoBERT / Vietnamese sentence embeddings + TF-IDF)
        │
        ▼
PHASE 7D: Taxonomy & Dataset Freezing
├── Step 6: Final decision on 11-class vs 12-class (+OTHER)
└── Step 7: Freeze final physical dataset & generate checksum-locked splits
        │
        ▼
PHASE 7E: True Multimodal Baseline Re-execution
├── Step 8: Re-run all 7 modality combinations on real features
└── Step 9: Re-compute baseline shift degradation metrics
        │
        ▼
PHASE 7F: Proposed UAD-Fusion Empirical Validation & Ablation
├── Step 10: Train UAD-Fusion on real multimodal features across 5 shift splits
├── Step 11: Execute complete ablation ladder (Models A through E)
└── Step 12: Multi-seed statistical significance testing & calibration analysis
        │
        ▼
PHASE 7G: Research Paper Writing & Artifact Release
```

---

## 18. Exact Project File Inventory

### Processed Metadata Catalogs (`data/processed/`)
- [master_metadata.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/master_metadata.csv) (8,738 rows)
- [final_trainable_metadata.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/final_trainable_metadata.csv) (5,416 rows)
- [final_12class_metadata.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/final_12class_metadata.csv) (5,514 rows)
- [trainable_metadata.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/trainable_metadata.csv) (5,416 rows)
- [trainable_physical_verified.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/trainable_physical_verified.csv) (5,416 rows)
- [manual_annotation_queue.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/manual_annotation_queue.csv) (3,322 rows)
- [rejected_records.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/rejected_records.csv) (321 rows)
- [recovery_blocked.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/recovery_blocked.csv) (5,876 rows)
- [audio_collection_queue.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/audio_collection_queue.csv) (5,416 rows)
- [cover_collection_queue.csv](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/data/processed/cover_collection_queue.csv) (5,416 rows)

### Primary Training & Benchmark Configs (`configs/`)
- [baseline.yaml](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/configs/baseline.yaml)
- [proposed.yaml](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/configs/proposed.yaml)

### Primary Scripts (`scripts/`)
- [train_baseline.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/train_baseline.py)
- [train_proposed.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/train_proposed.py)
- [create_final_splits.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/create_final_splits.py)
- [build_final_dataset.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/build_final_dataset.py)
- [phase6_stress_and_stats.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/phase6_stress_and_stats.py)
- [recover_lyrics.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/recover_lyrics.py)
- [recover_covers.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/recover_covers.py)
- [collect_physical_audio.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/collect_physical_audio.py)
- [collect_physical_covers.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/collect_physical_covers.py)
- [validate_final_dataset.py](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/scripts/validate_final_dataset.py)

### Primary Reports (`reports/`)
- [baseline_results.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/baseline_results.md)
- [proposed_method.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/proposed_method.md)
- [phase6_scientific_conclusion.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/phase6_scientific_conclusion.md)
- [phase6_data_completion_report.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/phase6_data_completion_report.md)
- [physical_modality_report.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/physical_modality_report.md)
- [final_artist_leakage_report.md](file:///c:/Workspace/Thuc%20hanh%20cac%20mon%20nam%203/N%C4%83m%204/NCKHRM-VMusic/RM-VMusic/reports/final_artist_leakage_report.md)
