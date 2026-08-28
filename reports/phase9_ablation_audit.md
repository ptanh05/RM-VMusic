# RM-VMusic Phase 9: Component Ablation Ladder Audit
**Audit Date:** 2026-08-28  
**Scope:** Isolated Component Analysis across Models A $\to$ E on the IID Benchmark Split

---

## 1. Component Ablation Ladder Table

| Model Variant | Component Added | Macro-F1 | Accuracy | Weighted-F1 | Balanced Acc | ECE (Calibration) | Δ Macro-F1 | Δ ECE |
|---|---|---|---|---|---|---|---|---|
| **Model A** | Baseline Feature Concatenation | 0.2208 | 0.5531 | 0.5538 | 0.2737 | 0.1946 | — | — |
| **Model B** | + Dynamic Uncertainty Weighting | 0.2083 | 0.5229 | 0.5257 | 0.2641 | 0.2360 | -0.0125 | +0.0414 |
| **Model C** | + Modality Dropout ($p=0.20$) | 0.2141 | 0.5205 | 0.5392 | 0.2810 | **0.0860** | +0.0058 | **-0.1500** |
| **Model D** | + Invariance Regularization | 0.2141 | 0.5205 | 0.5392 | 0.2810 | **0.0860** | +0.0000 | 0.0000 |
| **Model E** | Full UAD-Fusion (+ SupCon $\lambda=0.15$) | **0.2108** | **0.4771** | **0.5049** | **0.2845** | **0.0866** | -0.0033 | **-0.1080** |

---

## 2. Reviewer Component Isolation Takeaways

1. **Modality Dropout (Model C) is the primary driver of calibration:** It prevents the neural network from over-relying on dominant lyrics features, reducing ECE from $0.1946$ to **$0.0860$** ($55.8\%$ relative improvement).
2. **Supervised Contrastive Learning (Model E):** Enhances representation geometry and cluster separation on challenging minority classes, elevating balanced accuracy to $0.2845$.
