# RM-VMusic Phase 9: Verified Master Reproducibility Specification

This document provides the certified instructions to reproduce all empirical numbers, tables, checkpoints, and figures from scratch.

---

## 1. Certified Execution Environment

- **Python Version:** 3.10, 3.11, or 3.12 (Tested on 3.12.3 x64 Windows 11)
- **PyTorch Version:** PyTorch 2.5.1+cu121 (or CPU equivalent)
- **Dependencies:** `torch`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `Pillow`, `matplotlib`

---

## 2. One-Command Master Execution

To run the complete 9-stage pipeline from start to finish:
```bash
python scripts/run_all.py
```

### Modular Execution Breakdown:
1. **Audit & Manifests:** `python scripts/materialize_covers.py && python scripts/materialize_lyrics.py && python scripts/materialize_audio.py`
2. **Dataset & Splits:** `python scripts/build_12class_dataset.py && python scripts/create_final12_splits.py && python scripts/final12_leakage_audit.py`
3. **Feature Extraction:** `python scripts/extract_features.py`
4. **Master Multi-Seed Experiments:** `python scripts/run_master_experiments.py`
5. **Statistical CIs & Permutation Tests:** `python scripts/phase8_statistics.py`
6. **Publication Tables Export:** `python scripts/export_phase9_package.py`

---

## 3. Machine-Readable Artifacts

- **Publication CSVs:** `reports/paper/*.csv` (9 standardized tables).
- **Publication Figures:** `reports/figures/*.png` (12 high-resolution plots).
- **Consolidated Numerical Log:** `outputs/metrics/final_master_metrics.json`.
