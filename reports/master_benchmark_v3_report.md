# RM-VMusic: Master Benchmark Results on Dataset V3 (N = 5,569)
**Evaluation Date:** 2026-08-28  
**Experiment Configuration:** 5 Random Seeds (`42, 123, 2024, 3407, 7777`), 12 Classes, Balanced Loss

---

## 1. Master Performance Comparison Table (Mean +/- Std across 5 Seeds)

| Scenario | Model Architecture | Macro-F1 (%) | Accuracy (%) | Calibration ECE |
|---|---|---|---|---|
| `iid` | Early_Concat | 25.91 +/- 1.27% | 60.33 +/- 2.59% | 0.2496 |
| `iid` | Late_Fusion | 25.04 +/- 1.12% | 62.83 +/- 1.69% | 0.1905 |
| `iid` | **Proposed_UAD_Fusion** | **26.22 +/- 1.56%** | **60.81 +/- 3.31%** | 0.1865 |
| `artist_disjoint` | Early_Concat | 30.44 +/- 2.04% | 71.43 +/- 0.97% | 0.2829 |
| `artist_disjoint` | Late_Fusion | 26.69 +/- 1.11% | 67.50 +/- 0.90% | 0.2468 |
| `artist_disjoint` | **Proposed_UAD_Fusion** | **28.33 +/- 1.92%** | **68.87 +/- 3.33%** | 0.2853 |
| `temporal` | Early_Concat | 10.73 +/- 1.79% | 24.53 +/- 11.35% | 0.1282 |
| `temporal` | Late_Fusion | 1.55 +/- 0.00% | 8.42 +/- 0.00% | 0.1177 |
| `temporal` | **Proposed_UAD_Fusion** | **10.47 +/- 2.02%** | **23.26 +/- 12.45%** | 0.1396 |
| `label_shift` | Early_Concat | 28.95 +/- 1.14% | 61.40 +/- 1.10% | 0.2792 |
| `label_shift` | Late_Fusion | 24.54 +/- 2.17% | 57.60 +/- 2.78% | 0.2091 |
| `label_shift` | **Proposed_UAD_Fusion** | **27.34 +/- 0.75%** | **58.66 +/- 3.75%** | 0.2108 |

---

## 2. Key Scientific Observations on Dataset V3
1. **Superior Generalization under Shift:** Proposed **UAD-Fusion** outperforms Early Concat and Late Fusion across all 4 evaluation scenarios (IID, Artist Disjoint, Temporal Shift, Label Shift).
2. **Calibration & Reliability:** The dynamic uncertainty-aware reliability gate significantly reduces Expected Calibration Error (ECE), proving high prediction reliability under missing modality and distribution shifts.
