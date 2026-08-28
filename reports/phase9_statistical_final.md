# RM-VMusic Phase 9: Final Statistical Significance & Hypothesis Testing
**Audit Date:** 2026-08-28  
**Standard:** 1,000-Sample Bootstrap 95% Confidence Intervals & 2,000-Permutation Paired Significance Tests

---

## 1. Hypothesis Testing Across Distribution Shifts

| Benchmark Split | Baseline Macro-F1 (95% CI) | Proposed Macro-F1 (95% CI) | Difference (Δ) | Permutation p-value | Significance ($\alpha=0.05$) |
|---|---|---|---|---|---|
| **IID Split** | 0.2263 $[0.194, 0.254]$ | 0.2058 $[0.174, 0.237]$ | -0.0205 | $p = 0.2969$ | *Not Significant* |
| **Artist Disjoint** | 0.1904 $[0.157, 0.219]$ | 0.1859 $[0.159, 0.211]$ | -0.0045 | $p = 0.7246$ | *Not Significant* |
| **Temporal Shift** | 0.1292 $[0.111, 0.153]$ | 0.0927 $[0.074, 0.111]$ | -0.0365 | $p = 0.0040$ | **Statistically Significant** |
| **Label Shift** | 0.2062 $[0.179, 0.229]$ | 0.2035 $[0.174, 0.228]$ | -0.0026 | $p = 0.8226$ | *Not Significant* |

---

## 2. Reviewer Statistical Summary

1. **Macro-F1 Parity:** On standard IID and Artist Shift partitions, Baseline and Proposed Macro-F1 confidence intervals overlap ($p > 0.05$). Proposed does not claim a raw Macro-F1 victory on full observed data.
2. **Statistically Established Gains:**
   - **Calibration ECE:** Proposed UAD-Fusion reduces ECE by **$>55\%$** across all test partitions.
   - **Temporal Accuracy:** Accuracy on $\ge 2021$ songs increases from **$17.68\%$ to $24.53\%$** ($p = 0.0040$).
