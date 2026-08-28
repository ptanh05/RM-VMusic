# RM-VMusic Phase 6: Formal Audit of Phase 5 Experiments and Leakage Safeguards

This document provides the formal scientific audit verifying data isolation, leakage prevention, and experimental integrity for **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)**.

---

## 1. Audit Summary Matrix

| Audit Dimension | Target Criterion | Audit Status | Audit Details & Verification Evidence |
|-----------------|------------------|--------------|----------------------------------------|
| **1 Train Val Test Separation** | Strict Isolation & No Contamination | **PASS** | All 5 benchmark splits have mutually exclusive, disjoint partitions with 0 overlap. |
| **2 Artist Leakage** | Strict Isolation & No Contamination | **PASS** | Strict 0.00% artist leakage verified: 1,894 train artists vs 813 val/test artists (0 overlap). |
| **3 Label Leakage** | Strict Isolation & No Contamination | **PASS** | 100% of samples in trainable metadata belong to verified Tier A/B ground truth. 3,322 Tier C samples remain strictly isolated in manual_annotation_queue.csv. |
| **4 Temporal Leakage** | Strict Isolation & No Contamination | **PASS** | Strict temporal boundaries maintained (Train: <= 2018, Val: 2019-2020, Test: >= 2021). All 4,648 unverified release years excluded from evaluation. |
| **5 Duplicate Leakage** | Strict Isolation & No Contamination | **PASS** | 0.00% pairwise duplicate leakage across normalized (title, artist) strings between train and test partitions. |
| **6 Modality Leakage** | Strict Isolation & No Contamination | **PASS** | TF-IDF vocabulary and sublinear term frequency scalers are fitted strictly on the TRAIN partition in train_proposed.py and train_baseline.py; test lyrics are purely transformed. |
| **7 Test Set Contamination** | Strict Isolation & No Contamination | **PASS** | Class weights w_c are computed strictly from the TRAIN split (w_c = N_train / (C * N_train_c)). Early stopping checkpoints are selected solely based on Validation Macro-F1; Test set is touched only once during final evaluation. |
| **8 Checkpoint Selection** | Strict Isolation & No Contamination | **PASS** | Checkpoints in outputs/checkpoints/ are explicitly keyed by model variant and evaluated deterministically with fixed random seeds. |

---

## 2. Comprehensive Leakage Assessment

1. **Artist Independence**: `artist_disjoint.csv` strictly separates 1,894 training artists from 813 validation/test artists with **0.00% overlap**.
2. **Ground Truth Integrity**: 100% of samples evaluated in trainable metadata belong to Tier A or Tier B verified records. The 3,322 Tier C unannotated records remain completely isolated in `data/processed/manual_annotation_queue.csv`.
3. **Temporal Invariance**: The temporal evaluation set strictly enforces $T_{\text{train}} \le 2018$, $T_{\text{val}} = 2019-2020$, and $T_{\text{test}} \ge 2021$, with unverified records safely excluded from temporal evaluation.
4. **Feature & Preprocessing Isolation**: Modality feature vectorizers and class imbalance weights are computed strictly on the training partition without test-set leakage.

---
*Báo cáo kiểm toán Phase 6 tạo tự động bởi `scripts/phase6_audit.py` - RM-VMusic Pipeline.*
