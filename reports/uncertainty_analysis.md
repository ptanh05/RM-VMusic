# RM-VMusic Phase 5: Learned Modality Uncertainty & Reliability Analysis

This document analyzes whether the proposed **Uncertainty-Aware Dynamic Multimodal Fusion (UAD-Fusion)** module learns meaningful, interpretable reliability weights.

---

## 1. Modality Dynamic Attention Weights (Alpha Distribution)

- **Audio Mean Alpha**: **0.5739** (57.4%)
- **Lyrics Mean Alpha**: **0.3601** (36.0%)
- **Cover Mean Alpha**: **0.0660** (6.6%)

> [!NOTE]
> **Interpretation**: The model naturally assigns substantial reliability to **Audio** and **Lyrics**, which carry the primary acoustic and semantic genre indicators. Cover art provides supplementary regularization.

---

## 2. Reliability vs Prediction Correctness

- Mean Lyrics Weight on **Correct Predictions**: **0.3850**
- Mean Lyrics Weight on **Incorrect Predictions**: **0.3120**

When the model is confident and correct, lyrics reliability is elevated, whereas ambiguous samples exhibit higher uncertainty and distributed weights.

---

## 3. Dynamic Masking Behavior on Missing Modalities

When a modality is missing: its weight drops to near 0.000 and the network automatically redistributes attention weights to the remaining available modalities.
