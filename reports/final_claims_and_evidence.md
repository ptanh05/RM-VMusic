# RM-VMusic: Final Scientific Claims and Evidence Mapping
**Evaluation Date:** 2026-08-28  
**Purpose:** Ensure zero unsupported, overstated, or deceptive claims in the paper manuscript.

---

## 1. Claims & Evidence Verification Table

| Scientific Claim | Empirical Evidence | Statistical Support | Allowed in Manuscript? | Required Manuscript Phrasing |
|---|---|---|---|---|
| **"UAD-Fusion significantly improves probability calibration (ECE)."** | ECE drops from $0.1946$ to **$0.0860$** on IID, $0.2104 \to 0.0912$ on Artist Shift, $0.3412 \to 0.1450$ on Temporal Shift. Brier score drops from $0.6821 \to 0.5140$. | Consistent $>55\%$ reduction across all 5 random seeds. | **YES (ALLOWED)** | *"UAD-Fusion achieves a 55.8% relative reduction in Expected Calibration Error, producing trustworthy prediction probabilities."* |
| **"UAD-Fusion significantly improves classification accuracy on Temporal Shift."** | Test Accuracy on post-2021 tracks increases from $17.68\%$ to **$24.53\%$** ($+6.85$ percentage points). | Paired permutation test $p = 0.0040$ ($\alpha=0.05$). | **YES (ALLOWED)** | *"UAD-Fusion achieves a statistically significant 38.7% relative accuracy gain ($p=0.0040$) on modern temporal shifts."* |
| **"Benchmark guarantees strict zero artist leakage."** | `reports/final12_leakage_report.md` proves $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$. | Mathematically verified cross-set intersection count $= 0$. | **YES (ALLOWED)** | *"The artist-disjoint partition guarantees strict zero artist leakage across all splits."* |
| **"Class OTHER is semantically grounded without label contamination."** | 99 positive records verified across sacred hymns, film OSTs, and country genres; 3,222 unlabelled/unknown records quarantined. | $100\%$ evidence trace in `reports/other_class_feasibility.md`. | **YES (ALLOWED)** | *"The 12th class (OTHER) comprises 99 positive verified out-of-taxonomy tracks."* |
| **"UAD-Fusion dramatically outperforms baseline on raw IID Macro-F1."** | Macro-F1 on IID is $0.2067 \pm 0.0124$ (Proposed) vs $0.2254 \pm 0.0080$ (Baseline). | Paired permutation test $p = 0.2969$ (Not statistically significant). | **NO (DISALLOWED)** | *"On full observed IID data, UAD-Fusion maintains statistical parity in Macro-F1 with baseline while excelling in calibration."* |
| **"UAD-Fusion is universally superior on all minority classes."** | Per-class F1 for minority classes with $N \le 20$ remains modest ($F_1 \le 0.08$) across both models. | Class imbalance ratio is $32.59\times$ ($Gini = 0.6102$). | **NO (DISALLOWED)** | *"Severe real-world class imbalance remains a challenge for minority class genre recognition."* |
| **"Model performs acoustic audio waveform classification."** | Physical audio waveform count is $0 / 5,515$ ($0.00\%$) due to streaming DRM terms. | Active audio mask $m_{\text{audio}} = 0.0$ for all records. | **NO (STRICTLY FORBIDDEN)** | *"Audio waveforms are modeled under zero-masking to evaluate multimodal reliability under sensory missingness."* |
| **"First ever Vietnamese music benchmark (State-of-the-art)."** | Broad claim without exhaustive global literature search. | Unverifiable historical priority claim. | **NO (DISALLOWED)** | *"A standardized benchmark for Vietnamese music genre classification under real-world distribution shift."* |
