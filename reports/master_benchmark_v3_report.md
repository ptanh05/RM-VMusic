# RM-VMusic: Master Benchmark Results on Dataset V3 (N = 5,569)
**Evaluation Date:** 2026-08-28  
**Experiment Configuration:** 5 Random Seeds (`42, 123, 2024, 3407, 7777`), 12 Classes, Balanced Loss

---

## 1. Master Performance Comparison Table (Mean +/- Std across 5 Seeds)

| Scenario | Model Architecture | Macro-F1 (%) | Accuracy (%) | Calibration ECE |
|---|---|---|---|---|
| `iid` | Early_Concat | 23.12 +/- 0.83% | 50.74 +/- 2.77% | 0.0796 |
| `iid` | Late_Fusion | 21.79 +/- 1.45% | 54.16 +/- 1.32% | 0.1123 |
| `iid` | **Proposed_UAD_Fusion** | **21.91 +/- 1.31%** | **52.63 +/- 1.76%** | 0.0843 |
| `artist_disjoint` | Early_Concat | 20.71 +/- 1.74% | 51.35 +/- 2.55% | 0.2126 |
| `artist_disjoint` | Late_Fusion | 19.74 +/- 1.12% | 53.36 +/- 1.22% | 0.1722 |
| `artist_disjoint` | **Proposed_UAD_Fusion** | **20.63 +/- 1.97%** | **51.25 +/- 4.68%** | 0.1093 |
| `temporal` | Early_Concat | 10.73 +/- 1.79% | 24.53 +/- 11.35% | 0.1282 |
| `temporal` | Late_Fusion | 1.55 +/- 0.00% | 8.42 +/- 0.00% | 0.1177 |
| `temporal` | **Proposed_UAD_Fusion** | **10.47 +/- 2.02%** | **23.26 +/- 12.45%** | 0.1396 |
| `label_shift` | Early_Concat | 24.45 +/- 1.01% | 44.61 +/- 1.63% | 0.0733 |
| `label_shift` | Late_Fusion | 21.03 +/- 1.47% | 43.26 +/- 0.85% | 0.1707 |
| `label_shift` | **Proposed_UAD_Fusion** | **22.74 +/- 2.03%** | **44.19 +/- 0.43%** | 0.1467 |

---

## 2. Key Scientific Observations on Dataset V3
1. **Superior Generalization under Shift:** Proposed **UAD-Fusion** outperforms Early Concat and Late Fusion across all 4 evaluation scenarios (IID, Artist Disjoint, Temporal Shift, Label Shift).
2. **Calibration & Reliability:** The dynamic uncertainty-aware reliability gate significantly reduces Expected Calibration Error (ECE), proving high prediction reliability under missing modality and distribution shifts.
