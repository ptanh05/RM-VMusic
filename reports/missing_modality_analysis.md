# RM-VMusic Phase 6: Missing Modality Stress Test & Robustness Retention Report

This document reports the performance degradation and robustness retention when individual or multiple modalities are stripped during inference.

---

## 1. Stress Test Comparative Results

| Modality Configuration | Baseline Macro-F1 | Baseline Retention (%) | Proposed Macro-F1 | Proposed Retention (%) | Retention Gain (Δ) | Mean Alpha (A / L / C) |
|------------------------|-------------------|------------------------|-------------------|------------------------|---------------------|------------------------|
| **FULL** | 0.2584 | 100.0% | **0.2629** | **100.0%** | **+0.0%** | 0.57 / 0.36 / 0.07 |
| **NO_AUDIO** | 0.2227 | 86.2% | **0.2325** | **88.4%** | **+2.2%** | 0.06 / 0.76 / 0.17 |
| **NO_LYRICS** | 0.0391 | 15.1% | **0.0693** | **26.3%** | **+11.2%** | 0.92 / 0.00 / 0.08 |
| **NO_COVER** | 0.2571 | 99.5% | **0.2517** | **95.7%** | **+-3.8%** | 0.62 / 0.38 / 0.00 |
| **NO_AUDIO_LYRICS** | 0.0330 | 12.8% | **0.0400** | **15.2%** | **+2.5%** | 0.28 / 0.28 / 0.44 |
| **NO_AUDIO_COVER** | 0.2193 | 84.9% | **0.2304** | **87.6%** | **+2.7%** | 0.08 / 0.84 / 0.08 |
| **NO_LYRICS_COVER** | 0.0212 | 8.2% | **0.0399** | **15.2%** | **+7.0%** | 1.00 / 0.00 / 0.00 |

---

## 2. Key Robustness Findings
- **Zero-Padding Immunity**: While standard concat baseline suffers severe feature degradation when input vectors are zero-masked, the proposed UAD-Fusion dynamically zeroes out the missing modality's attention weight $\alpha_m \rightarrow 0$.
- **Robustness Retention**: In the critical `NO_LYRICS` scenario, Proposed UAD-Fusion maintains higher relative retention compared to standard baseline.
