# RM-VMusic Phase 9: Modality Baseline Audit Report
**Audit Date:** 2026-08-28  
**Scope:** Re-Execution & Performance Verification of All 7 Modality Combinations

---

## 1. Modality Baseline Results Table (IID Benchmark Split)

| Modality Combination | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Operational & Physical Interpretation |
|---|---|---|---|---|---|
| `audio_only`* | 0.5495 | 0.0591 | 0.3898 | 0.0833 | Degenerates to majority prior under zero-masking ($mask=0.0$) |
| `cover_only` | 0.0894 | 0.0297 | 0.0948 | 0.0943 | Weak visual color signal on 16.36% coverage |
| `lyrics_only` | 0.4771 | 0.2088 | 0.5083 | 0.2691 | Dominant unimodal predictor from 74.65% physical lyrics |
| `audio_lyrics` | 0.4855 | 0.2289 | 0.5215 | 0.2886 | Lyrics-driven representation with zero-masked audio |
| `audio_cover` | 0.0495 | 0.0310 | 0.0417 | 0.0966 | Weak combination due to audio absence and cover sparsity |
| `lyrics_cover` | 0.5254 | 0.2009 | 0.5358 | 0.2467 | Dual physical modality (Lyrics + Decoded Cover Images) |
| `audio_lyrics_cover` | **0.5435** | **0.2396** | **0.5625** | **0.2947** | Full Multimodal Concatenation (Baseline Reference) |

*\*Note: `audio_only` achieves 54.95% accuracy by predicting the dominant `POP_BALLAD` class due to zero audio waveforms on disk.*

---

## 2. Baseline Fairness Audit

- **Splits:** Identical train (`3,860`), validation (`827`), and test (`828`) sets across all models.
- **Seeds:** Identical random seed sequence (`[42, 123, 2024, 3407, 7777]`).
- **Loss Function:** Identical class-weighted cross-entropy loss computed on the train partition.
- **Early Stopping:** Identical patience of 8 epochs on validation Macro-F1.
- **Audit Finding:** **PASSED (100% fair and standardized baseline setup)**.
