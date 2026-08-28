# RM-VMusic Phase 9: 12-Class Taxonomy & Class Imbalance Analysis
**Audit Standard:** ISMIR / ICASSP Reviewer Standards  
**Audit Date:** 2026-08-28  
**Dataset Scope:** $N=5,515$ Tracks across 12 Vietnamese Music Genres

---

## 1. Class Frequency & Imbalance Statistics

| Index | Genre Class | Trainable Samples | Relative % | Unique Artists | Test Samples | Imbalance Ratio vs. Min |
|---|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | 455 | **32.59x** |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | 121 | **8.68x** |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | 43 | **3.09x** |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | 33 | **2.38x** |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | 30 | **2.15x** |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | 29 | **2.08x** |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | 26 | **1.83x** |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | 22 | **1.56x** |
| 8 | `ROCK` | 137 | 2.48% | 20 | 21 | **1.47x** |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | 20 | **1.42x** |
| 10 | `OTHER` | 99 | 1.80% | 54 | 15 | **1.06x** |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | 14 | **1.00x** (Reference) |

---

## 2. Statistical Dispersion & Concentration Metrics

- **Gini Concentration Index:** **$0.6102$** (Signifies high natural concentration in dominant genres).
- **Shannon Entropy:** **$2.3885$ bits** (Maximum theoretical uniform entropy for 12 classes is $\log_2(12) \approx 3.585$ bits).
- **Dominance Share:** Top 2 genres (`POP_BALLAD` + `BOLERO_TRUTINH`) account for **$69.59\%$** of all tracks.
- **Scientific Implication:** This heavy skew reflects the real-world commercial Vietnamese music market. Consequently, the research must employ **Balanced Cross-Entropy** and report **Macro-F1** and **Balanced Accuracy** to penalize majority-class bias.
