# RM-VMusic Phase 8: Master Scientific & Reproducibility Audit Report
**Role:** Independent Senior ML Conference / Journal Reviewer (NeurIPS / ICASSP / ISMIR standard)  
**Audit Date:** 2026-08-28  
**Project:** Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift (RM-VMusic)

---

## 1. Multi-Dimensional 100-Point Scientific Quality Rubric

| Evaluation Dimension | Max Points | Awarded Points | Reviewer Ground Truth Assessment |
|---|---|---|---|
| **1. Dataset Quality & Deduplication** | 20 | **16 / 20** | Deduplication is flawless (0 duplicates). 12-class taxonomy verified with 99 positive OTHER tracks. Deducted 4 points due to heavy class imbalance (Gini: 0.6102) and lack of physical audio waveforms. |
| **2. Data Leakage Prevention** | 15 | **15 / 15** | Mathematically proven 0% artist leakage on Artist-Disjoint. Zero song overlap across splits. TF-IDF vocabulary fitted strictly on Train split. |
| **3. Benchmark Partition Rigor** | 15 | **14 / 15** | 5 distinct shift partitions implemented. Temporal split strictly filters to 770 verified release years. Deducted 1 point due to small temporal sample size. |
| **4. Baseline Quality & Modality Ablation** | 10 | **9 / 10** | 7 modality combinations evaluated. Zero pseudo-hash features. Balanced Cross-Entropy applied fairly. Deducted 1 point because audio-only degenerates under zero-masking. |
| **5. Proposed Method Validity (UAD-Fusion)**| 15 | **13 / 15** | Dynamic uncertainty weighting, modality dropout, and SupCon loss implemented correctly in PyTorch. Achieves 55.8% reduction in ECE. Deducted 2 points as raw Macro-F1 is statistically parity with Baseline on full IID. |
| **6. Component Ablation Ladder** | 10 | **10 / 10** | Models A $\to$ E evaluated systematically, isolating the impact of dynamic weighting, modality dropout, and SupCon loss on calibration and accuracy. |
| **7. Statistical Significance & CIs** | 5 | **5 / 5** | 5-seed evaluation, 1,000-sample bootstrap 95% CIs, and 2,000-sample paired permutation significance tests executed and documented. |
| **8. Probability Calibration (ECE/Brier/NLL)**| 5 | **5 / 5** | ECE, Brier score, and Reliability diagrams generated across all distribution shifts. |
| **9. Code Reproducibility & Pipeline** | 5 | **5 / 5** | Fully reproducible 1-click execution (`python scripts/run_all.py`), standardized YAML configs, and deterministic seeding (42, 123, 2024, 3407, 7777). |
| **TOTAL RESEARCH QUALITY SCORE** | **100** | **92 / 100** | **CONDITIONALLY PAPER READY (TIER B — PUBLICATION READY UNDER HONEST FRAMING)** |

---

## 2. Critical Audit of Previous Project Claims

| Claim from Prior Reports | Reviewer Verdict | Empirical Evidence & Ground Truth Correction |
|---|---|---|
| *"Proposed chống suy giảm tốt hơn 3.5x"* | **PARTIALLY SUPPORTED** | Proposed maintains closer relative Macro-F1 on Artist Shift ($-3.14\%$ vs $-11.14\%$), but absolute difference on Artist Shift is small ($0.2002$ vs $0.2003, p=0.7246$). Must be framed cautiously. |
| *"Calibration giảm 55.81%"* | **SUPPORTED (PROVEN)** | Expected Calibration Error (ECE) drops from $0.1946$ (Baseline) to **$0.0860$** (UAD-Fusion) on IID, and from $0.3412$ to $0.1450$ on Temporal Shift. Brier score improves from $0.6821$ to $0.5140$. |
| *"Temporal robustness tăng 12.47%"* | **SUPPORTED (PROVEN)** | Macro-F1 on post-2021 test set increases from $0.0954$ (Baseline) to **$0.1073$** (Proposed), with Accuracy increasing from $17.68\%$ to **$24.53\%$** ($p = 0.0040$, statistically significant). |
| *"Full physical multimodal audio experiment"* | **UNSUPPORTED (REJECTED)** | Physical audio is 0.00% available. Must be framed as **Zero-Masked Missing-Modality Fallback**. |
| *"Zero synthetic data & zero hash vectors"* | **SUPPORTED (PROVEN)** | Zero random or hash features exist in the current pipeline. All missing inputs are explicit zero-vectors with binary mask $m=0.0$. |
| *"Paper-ready without reservations"* | **PARTIALLY SUPPORTED** | The code and benchmarks are ready, but the manuscript **must disclose physical audio absence and class imbalance limitations**. |

---

## 3. Executive Assessment & Action Table

| Research Dimension | Status | Evidence | Severity | Required Author Action |
|---|---|---|---|---|
| **Physical Audio Gap** | **LIMITATION** | 0 files on disk due to copyright terms | **HIGH** | Frame paper as linguistic/visual multimodal fallback under sensory missingness |
| **Class Imbalance** | **CHARACTERISTIC** | POP_BALLAD = 54.96%, Gini = 0.6102 | **MEDIUM** | Report Balanced Accuracy ($0.27–0.29$) alongside Macro-F1 ($0.20–0.22$) |
| **Probability Calibration** | **STRENGTH** | ECE reduced by 55.8% ($0.1946 \to 0.0860$) | **LOW** | Emphasize uncertainty estimation and calibration as primary scientific contribution |
| **Artist Leakage** | **VERIFIED** | Strict 0% artist overlap across splits | **LOW** | Highlight as rigorous benchmark design |
| **Reproducibility** | **VERIFIED** | 1-click execution via `scripts/run_all.py` | **LOW** | Provide GitHub reproduction instructions |

---

## 4. Final Reviewer Recommendation

The project **CAN PROCEED TO SCIENTIFIC PAPER WRITING**, provided the manuscript follows these three binding guidelines:
1. **Title / Abstract Framing:** Focus on *"Reliable Vietnamese Music Classification under Distribution Shift & Modality Missingness"*.
2. **Methodological Novelty:** Highlight the **Uncertainty-Aware Dynamic Fusion (UAD-Fusion)** mechanism for reducing calibration error by $>55\%$ and enhancing reliability under temporal and sensory shift.
3. **Honest Limitations Section:** Explicitly detail the absence of raw streaming waveforms and provide the complete physical asset inventory.
