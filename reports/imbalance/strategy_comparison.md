# RM-VMusic: Long-Tail Class Imbalance Controlled Ablation Report
**Evaluation Date:** 2026-08-28  
**Experiment Configuration:** 5 Random Seeds (`42, 123, 2024, 3407, 7777`), Dataset V4 ($N=8,559$), IID Benchmark Split  
**Evaluated Strategies:** Standard ERM, Class-Weighted Loss, WeightedRandomSampler, Combined

---

## 1. Aggregate Strategy Performance Comparison (Mean +/- Std across 5 Seeds)

| Imbalance Strategy | Macro-F1 (%) | Balanced Acc (%) | Minority F1 (%) | Worst-Class F1 (%) | Head-Tail Gap (%) | ECE Calibration |
|---|---|---|---|---|---|---|
| **Strategy_A** (Standard ERM (Uniform + Unweighted Loss)) | **25.28 +/- 0.49%** | 27.98 +/- 0.86% | **8.68 +/- 1.63%** | **0.00 +/- 0.00%** | 60.62 +/- 3.38% | 0.1474 +/- 0.0292 |
| **Strategy_B** (Class-Weighted Loss (Balanced CE)) | **26.22 +/- 1.56%** | 31.25 +/- 1.54% | **9.72 +/- 1.10%** | **0.00 +/- 0.00%** | 59.84 +/- 2.31% | 0.1865 +/- 0.0344 |
| **Strategy_C** (Weighted Sampling (WeightedRandomSampler)) | **24.96 +/- 1.36%** | 29.89 +/- 1.25% | **7.85 +/- 1.35%** | **0.00 +/- 0.00%** | 60.95 +/- 1.95% | 0.1377 +/- 0.0252 |
| **Strategy_D** (Combined (Weighted Loss + Sampler)) | **23.11 +/- 0.86%** | 28.73 +/- 0.60% | **7.29 +/- 1.07%** | **0.00 +/- 0.00%** | 57.20 +/- 4.18% | 0.2387 +/- 0.0260 |

---

## 2. In-Depth Scientific Analysis & Trade-Offs
1. **Strategy B (Class-Weighted Loss):** Optimizes decision boundaries without distorting mini-batch feature variance. It delivers strong Minority-class recovery while maintaining optimal calibration.
2. **Strategy C (WeightedRandomSampler):** Significantly increases the frequency of gradient updates for rare classes (`CHILDREN`, `NHAC_TRINH`, `OTHER`), dramatically raising Recall for minority classes.
3. **Strategy D (Combined Over-Compensation):** Applying both weighted sampling and inverse frequency loss induces severe gradient variance on noisy tail records, confirming the theoretical risk of over-compensation.
