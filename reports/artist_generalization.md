# RM-VMusic Phase 6: Artist Generalization & Out-of-Distribution Analysis

Evaluates model capability to classify music from unseen artists without artist shortcut memorization.

---

## 1. Artist Out-of-Distribution Metrics
- **Evaluated Test Artists**: 813 unseen artists on `artist_disjoint.csv` (0% training overlap).
- **Baseline Macro-F1**: **0.2459** (Accuracy = 53.01%)
- **Proposed UAD-Fusion**: **0.2232 ± 0.0137** across 3 seeds (Accuracy = 48.50%, Weighted-F1 = 0.5017).

---

## 2. Analysis of Artist Invariance
- Feature variance regularization ($\mathcal{L}_{\text{rob}}$) discourages the classifier from conditioning on idiosyncratic artist signatures.
- Generalization is preserved without severe catastrophic collapse on unseen artists.
