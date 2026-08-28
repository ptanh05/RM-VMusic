# RM-VMusic: Comprehensive Senior ML Reviewer Report
**Reviewer Role:** Meta-Reviewer / Senior Reviewer for ISMIR / ICASSP / ACM Multimedia  
**Evaluation Date:** 2026-08-28  
**Manuscript Topic:** Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift & Modality Missingness

---

## 1. Summary of the Work
The paper introduces **RM-VMusic**, a curated benchmark for Vietnamese music genre classification consisting of 5,515 trainable tracks across 12 genre classes. The authors evaluate multimodal classification models across five distribution shift partitions (IID, Artist-Disjoint with 0% leakage, Temporal shift across verified release years 1967–2026, Label Shift, and an 11-step Missing Modality stress curve). They propose **UAD-Fusion**, an Uncertainty-Aware Dynamic Multimodal Fusion network designed to estimate modality reliability, apply modality dropout, and optimize supervised contrastive losses to combat sensory deprivation.

---

## 2. Key Strengths
1. **Uncompromising Data Hygiene & Leakage Prevention:** The benchmark provides mathematical proof of 0% artist leakage on artist-disjoint splits, 0 duplicate tracks, and strictly isolates vocabulary fitting to the training split.
2. **Elimination of Pseudo-Features:** Unlike earlier experimental baselines that relied on heuristic hash embeddings, all models run strictly on physical text (TF-IDF), decoded JPEG images, and explicit zero-masking for missing modalities.
3. **Remarkable Probability Calibration Gains:** UAD-Fusion achieves a consistent $>55\%$ reduction in Expected Calibration Error (ECE from $0.1946 \to 0.0860$), ensuring well-calibrated confidence estimates under missing modalities.
4. **Statistically Significant Temporal Shift Robustness:** Accuracy on post-2021 modern songs improves significantly from $17.68\%$ to $24.53\%$ ($p = 0.0040$).
5. **Outstanding Reproducibility:** A single automated script (`scripts/run_all.py`) and clean publication tables (`reports/paper/*.csv`) allow 1-click verification of all results.

---

## 3. Major Weaknesses & Limitations
1. **Total Absence of Physical Audio Waveforms (0.00% Coverage):** Due to copyright boundaries and streaming CDN HMAC token expiration, audio is completely zero-masked. The paper cannot claim acoustic feature modeling and must be strictly positioned as a linguistic/visual multimodal fallback study.
2. **Severe Class Imbalance ($Gini = 0.6102$):** Dominant `POP_BALLAD` ($54.96\%$) creates severe difficulty for extreme minority classes ($N \le 20$ in test set like `OTHER`, `CHILDREN`, `ROCK`), where F1 scores remain low across all models.
3. **Macro-F1 Statistical Parity on IID Data:** On full observed IID data, UAD-Fusion does not achieve a statistically significant Macro-F1 advantage over baseline concatenation ($p = 0.2969$). Its primary contribution is calibration and temporal robustness.

---

## 4. Minor Weaknesses
1. The temporal shift benchmark isolates 770 verified release year tracks (13.96% of catalog). While statistically clean, it represents a subset of the full dataset.
2. Lyrics representation utilizes classical TF-IDF (5,000 n-grams) rather than deep pretrained contextual transformer embeddings (e.g., PhoBERT).

---

## 5. Methodological & Statistical Assessment
- **Statistical Testing:** 5-seed evaluation, 1,000-sample bootstrap 95% CIs, and paired permutation tests are executed with proper sample-level pairing.
- **Fairness:** Baseline and Proposed architectures share identical seeds, splits, class weights, and early stopping patience.

---

## 6. Required Revisions Before Camera-Ready
1. **Mandatory Abstract / Title Reframing:** Title must emphasize *"Reliability, Modality Missingness, and Distribution Shift"*.
2. **Explicit Audio Disclosure:** Prominently disclose the 0.00% physical audio coverage in the Abstract and Dataset sections.
3. **Balanced Metric Presentation:** Always present Balanced Accuracy ($0.27–0.29$) alongside Macro-F1 ($0.20–0.22$) to properly represent class imbalance dynamics.

---

## 7. Final Reviewer Recommendation: **ACCEPT WITH MINOR REVISIONS (CONDITIONALLY PAPER READY)**
The paper offers a valuable, transparent, and reproducible contribution to the music information retrieval and distribution-shift literature, provided the authors maintain absolute honesty regarding audio absence and class imbalance.
