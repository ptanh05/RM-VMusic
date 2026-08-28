# RM-VMusic Phase 9: Comprehensive Claims Audit & Manuscript Boundaries
**Audit Date:** 2026-08-28  
**Scope:** Verification of All Scientific Claims against Empirical Data and Physical Assets

---

## 1. Categorization of Claims

### A. VERIFIED CLAIMS (Direct Empirical Evidence Available)
- ✅ *"Deduplication is strict with 0 duplicate song IDs, titles, or sources."* (Proven in `reports/final_dedup_report.md`).
- ✅ *"Artist Disjoint benchmark achieves strict 0% artist leakage."* (Proven in `reports/final12_leakage_report.md`).
- ✅ *"UAD-Fusion reduces Expected Calibration Error (ECE) by >55% across distribution shifts."* (Proven in `outputs/metrics/final_master_metrics.json`).
- ✅ *"Zero deterministic hash vectors or pseudo-features exist in the final pipeline."* (Proven in `scripts/extract_features.py`).
- ✅ *"Class OTHER is semantically defined with 99 verified out-of-taxonomy tracks."* (Proven in `reports/other_class_feasibility.md`).

---

### B. PARTIALLY SUPPORTED CLAIMS (Restricted Scope Required)
- ⚠️ *"Proposed method achieves superior shift robustness."* $\to$ **Correction:** Proposed maintains closer relative stability and higher Accuracy on Temporal shift, but Macro-F1 on full IID is statistically comparable ($p=0.2969$).
- ⚠️ *"Temporal shift benchmark spans multi-decade Vietnamese music history."* $\to$ **Correction:** Limited to the subset of 770 verified release year tracks (1967–2026).

---

### C. UNSUPPORTED CLAIMS (Do Not Include in Manuscript)
- ❌ *"Audio-based classification achieves high multimodal accuracy."* (Unsupported: Physical audio is 0.00% available).
- ❌ *"Proposed UAD-Fusion dramatically outperforms baseline on minority classes."* (Unsupported: Minority class F1 scores remain low across all models due to natural 32.59x market imbalance).

---

### D. FORBIDDEN CLAIMS (Strictly Banned)
- 🚫 **"First ever Vietnamese music benchmark"** (Cannot claim without exhaustive global literature proof; rephrase to *"A standardized benchmark for Vietnamese music genre classification under distribution shift"*).
- 🚫 **"State-of-the-art acoustic genre classification"** (Physical audio waveforms are absent).
