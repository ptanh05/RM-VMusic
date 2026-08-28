# RM-VMusic Phase 9: Final Decision & Next Actions
**Evaluation Date:** 2026-08-28  
**Final Scientific Verdict:** **B. PAPER READY WITH MAJOR CAVEATS**

---

## 1. Justification for Status B ("Paper Ready with Major Caveats")

The RM-VMusic benchmark is mathematically clean, deduplicated, and 100% reproducible. However, it cannot be classified as "Unconditional Category A" because:
1. **Physical Audio is 0.00% Available:** Due to legal boundaries and Zing MP3 expiring tokens, raw audio waveforms are absent. The paper must be explicitly scoped as **"Multimodal Reliability under Missing Modalities and Shift"**, rather than an acoustic classification study.
2. **Class Imbalance is Severe ($Gini = 0.6102$):** Dominant `POP_BALLAD` ($54.96\%$) causes minority class F1 scores to remain modest ($F_1 \le 0.08$), requiring Balanced Accuracy and Weighted-F1 to be reported alongside Macro-F1.
3. **Primary Contribution is Calibration & Fallback:** The core empirical strength of UAD-Fusion is a **$55.8\%$ reduction in Expected Calibration Error (ECE from $0.1946 \to 0.0860$)** and superior accuracy on temporal shift, rather than a raw Macro-F1 victory on full observed IID data.

---

## 2. EXACT NEXT ACTIONS (Prioritized Roadmap)

### 🔴 P0 (Mandatory Before Manuscript Submission)
1. **Manuscript Framing Alignment:** Title the paper *"RM-VMusic: A Benchmark for Reliable Vietnamese Music Classification under Real-World Distribution Shift and Modality Missingness"*.
2. **Explicit Modality Disclosure:** Clearly disclose in the Abstract and Dataset section that raw audio streams are modeled under zero-masking conditions ($mask = 0.0$) due to streaming DRM constraints.
3. **Use Official Paper Tables:** Integrate the standardized CSV tables from `reports/paper/` directly into the LaTeX manuscript.

### 🟠 P1 (Highly Recommended for Rebuttal / Extended Journal Version)
4. **Ingest Open-Source Vietnamese Audio:** If copyright-free CC-BY Vietnamese music recordings become available in the future, ingest them to benchmark acoustic encoders.
5. **Pretrained PhoBERT Lyrics Embeddings:** Compare classical TF-IDF (5,000 n-grams) against pretrained PhoBERT embeddings to explore deeper contextual semantics.

### 🟡 P2 (Optional Engineering Polish)
6. **Web Demonstration Dashboard:** Package UAD-Fusion into a lightweight Streamlit/Gradio web demo for interactive uncertainty visualization.
7. **Zenodo Data Archival:** Upload metadata, splits, and features to Zenodo/HuggingFace with a persistent DOI upon paper acceptance.
