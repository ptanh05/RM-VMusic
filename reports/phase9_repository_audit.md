# RM-VMusic Phase 9: Comprehensive Repository Architecture & Codebase Audit
**Reviewer Role:** Senior Reviewer (ISMIR / ICASSP / ACM Multimedia Standard)  
**Audit Date:** 2026-08-28  
**Repository:** `RM-VMusic` (*Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift*)

---

## 1. Directory Structure & Asset Inventory

```text
RM-VMusic/
├── configs/
│   ├── baseline.yaml                     # Baseline hyperparameters
│   ├── proposed.yaml                     # Proposed UAD-Fusion hyperparameters
│   └── final_experiment.yaml             # Master consolidated benchmark specification
│
├── data/
│   ├── raw/                              # Immutable source downloads (sunbv56, vietlyrics)
│   ├── processed/
│   │   ├── master_metadata.csv           # Full catalog (8,738 tracks)
│   │   ├── final_12class_metadata.csv    # Final 12-class trainable dataset (5,515 tracks)
│   │   ├── audio_manifest.csv            # Physical audio materialization audit (0 files)
│   │   ├── cover_manifest.csv            # Cover art materialization audit (1,445 files)
│   │   ├── lyrics_manifest.csv           # Physical lyrics audit (4,117 files)
│   │   └── final12_modality_matrix.csv   # Per-track physical modality availability matrix
│   ├── audio/                            # Physical audio directory (0 files, 0.00%)
│   ├── covers/                           # 1,445 physical JPEG images on disk
│   ├── lyrics/                           # 4,117 physical text files on disk
│   ├── features/                         # Physical feature tensors (5000-dim TF-IDF, 512-dim visual)
│   └── splits/                           # 5 Distribution Shift splits (IID, Artist, Temporal, Label, Missing)
│
├── scripts/
│   ├── build_12class_dataset.py          # Builds final 12-class metadata with OTHER
│   ├── create_final12_splits.py          # Generates 5 benchmark partitions
│   ├── final12_leakage_audit.py          # Validates zero duplicate IDs and 0% artist leakage
│   ├── extract_features.py               # Extracts real TF-IDF and visual moments (no pseudo-features)
│   ├── train_physical_baselines.py       # Trains 7 baseline modality configurations
│   ├── train_proposed.py                 # UAD-Fusion PyTorch architecture & SupCon loss
│   ├── run_master_experiments.py         # Multi-seed runner across all 5 shifts
│   ├── phase8_statistics.py              # Bootstrap CIs & permutation significance testing
│   ├── export_phase9_package.py          # Exports paper data package (9 CSVs)
│   └── run_all.py                        # 1-click end-to-end master reproduction script
│
├── outputs/
│   ├── checkpoints/                      # Trained model weights on physical features
│   └── metrics/final_master_metrics.json # Full numerical metrics log
│
├── reports/
│   ├── figures/                          # 12 high-resolution publication PNG figures
│   ├── paper/                            # 9 publication-ready CSV tables & README
│   └── *.md                              # Detailed audit and evaluation reports
│
└── docs/
    ├── final_dataset_card.md             # Dataset card & provenance
    ├── phase7b_methodology.md            # Technical methodology
    └── reproducibility.md                # Reproduction instructions
```

---

## 2. Implementation vs. Documentation Discrepancy Audit

| Documented Claim / Specification | Implementation Code Check | Actual Ground Truth Finding | Reviewer Verdict |
|---|---|---|---|
| **Taxonomy Scope** | 12 classes in `build_12class_dataset.py` | Exactly 12 classes; 99 samples in `OTHER` | **VERIFIED (MATCH)** |
| **Physical Audio** | 0 files in `data/audio/` | $0$ physical files; zero-masked in `extract_features.py` | **VERIFIED (MATCH)** |
| **Physical Covers** | 1,445 files on disk, 902 in trainable | 1,445 files in `data/covers/`, 902 active masks | **VERIFIED (MATCH)** |
| **Physical Lyrics** | 4,117 files on disk | 4,117 text files in `data/lyrics/`, 4,117 active masks | **VERIFIED (MATCH)** |
| **Pseudo-Features** | Banned in `extract_features.py` | No hash or random features found | **VERIFIED (MATCH)** |
| **Artist Leakage** | 0% in `final12_leakage_audit.py` | $\text{Tr} \cap \text{Va} = 0, \text{Tr} \cap \text{Te} = 0, \text{Va} \cap \text{Te} = 0$ | **VERIFIED (MATCH)** |
| **TF-IDF Vocabulary** | Fitted on Train only | `vectorizer.fit(train_texts)` strictly in `extract_features.py` | **VERIFIED (MATCH)** |
