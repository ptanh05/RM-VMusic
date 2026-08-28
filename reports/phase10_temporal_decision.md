# RM-VMusic Phase 10: Temporal Benchmark Scientific Decision & Roadmap
**Evaluation Date:** 2026-08-28  
**Scientific Classification:** **C. SELECTION-BIASED TEMPORAL BENCHMARK (VALID BUT MUST BE EXPLICITLY POSITIONED AS A CURATED SUBSET EVALUATION)**

---

## 1. Scientific Justification

The temporal benchmark ($N=770$) cannot be claimed as an unbiased chronological sample of the full 5,515-track Vietnamese music ecosystem because:
1. **Selection Bias is Proven:** Year metadata was systematically missing for $98.9\%$ of `POP_BALLAD` and `BOLERO_TRUTINH` tracks in the raw crawls, while being richly populated for `ROCK` ($87.6\%$) and `RB_SOUL` ($84.9\%$).
2. **Structural Class Missingness in Test Set:** `NHAC_TRINH` ($N=0$) and `CHILDREN` ($N=0$) are absent in modern tracks ($\ge 2021$), while `OTHER` ($N=0$) is absent in training ($\le 2018$).

---

## 2. Definitive Action Decisions

- **Dataset Rebuild Required?** **NO.** The core dataset ($N=5,515$) is clean, deduplicated, and sound. Imputing release years without ground truth would violate scientific honesty.
- **Temporal Split Rebuild Required?** **NO.** The current chronological thresholds ($\le 2018$, $2019-2020$, $\ge 2021$) reflect the standard pre-pandemic vs post-pandemic temporal evaluation protocol.
- **Release-Year Metadata Improvement Required?** **YES (For future extensions).** Future data collection should target explicit discography release dates for mainstream Pop and Bolero tracks.
- **Model Retraining Required?** **NO.** Existing multi-seed benchmark results accurately reflect this data reality.

---

## 3. Mandatory Manuscript Reporting Guidelines

1. **Explicit 10-Class Disclosure:** In the Temporal Shift section, state clearly:
   > *"Due to historical genre life cycles and metadata availability on verified release year tracks ($N=770$), the temporal test set ($\ge 2021$) comprises 10 active genre classes (190 tracks), with historical genres such as Nhạc Trịnh and Children's songs absent in the post-2021 test partition."*
2. **Dual Metric Presentation:** Report both the standard 12-class Macro-F1 ($0.1073$) and note the impact of zero-support classes.
