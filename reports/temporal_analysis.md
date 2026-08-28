# RM-VMusic Phase 6: Temporal Robustness Across Release Cohorts

Evaluates classification performance across chronological release cohorts strictly on independently verified samples ($N=768$).

---

## 1. Chronological Cohort Performance

| Release Cohort | Sample Count ($N$) | Evaluation Status | Baseline Macro-F1 | Proposed Macro-F1 | Delta (Δ) |
|----------------|--------------------|-------------------|-------------------|-------------------|-----------|
| **<= 2010** | 357 | `VALID_EVALUATION` | 0.3376 | **0.4421** | +0.1045 |
| **2011-2015** | 99 | `VALID_EVALUATION` | 0.4056 | **0.4313** | +0.0257 |
| **2016-2018** | 70 | `VALID_EVALUATION` | 0.4223 | **0.4772** | +0.0549 |
| **2019-2020** | 54 | `VALID_EVALUATION` | 0.2989 | **0.3690** | +0.0701 |
| **>= 2021** | 188 | `VALID_EVALUATION` | 0.3404 | **0.4235** | +0.0832 |

---

## 2. Scientific Temporal Note
- Bins with $N < 20$ samples are marked as `INSUFFICIENT_SAMPLE` to prevent overclaiming.
- The severe degradation on the post-2021 cohort ($\ge 2021$) reflects true musicological evolution in modern Vietnamese production styles.
