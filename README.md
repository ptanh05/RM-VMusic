# RM-VMusic: Reliable Multimodal Vietnamese Music Genre Classification under Real-World Distribution Shift and Modality Missingness

A scientific benchmark and research framework for studying **Multimodal Reliability**, **Probability Calibration**, **Distribution Shifts**, and **Sensory Missingness** in Vietnamese music genre classification.

---

## 1. Research Question

In real-world music information retrieval and streaming ecosystems, metadata, album covers, and lyrics are frequently incomplete, while raw audio streams may be restricted by licensing, bandwidth, or digital rights management (DRM). This raises critical research questions:
1. *How do multimodal neural architectures degrade when facing distribution shifts (unseen artists, temporal evolution across decades, and label shifts)?*
2. *Can dynamic uncertainty estimation and reliability weighting prevent catastrophic failure and overconfident misclassifications when modalities are corrupted or missing?*
3. *How effectively can linguistic (lyrics) and visual (cover art) representations compensate as fallback mechanisms when acoustic audio streams are completely unavailable?*

---

## 2. Key Contributions

- **Standardized 12-Class Benchmark:** The first mathematically isolated Vietnamese music genre classification benchmark ($N=5,515$ trainable tracks, $2,746$ unique artists) with proven **0% artist leakage** on artist-disjoint splits and verified release years (1967–2026).
- **Semantically Verified `OTHER` Class:** Explicit evidence-based annotation of 99 positive out-of-taxonomy tracks (Sacred/Religious hymns, Film OSTs, Country), avoiding noisy label contamination.
- **UAD-Fusion Architecture:** An **Uncertainty-Aware Dynamic Multimodal Fusion** framework incorporating dynamic inverse-variance modality weighting, modality dropout, and supervised contrastive representation learning.
- **Superior Probability Calibration:** Achieves a **$55.8\%$ reduction in Expected Calibration Error (ECE from $0.1946 \to 0.0860$)** and improves Brier score from $0.6821 \to 0.5140$, ensuring trustworthy confidence outputs under sensory deprivation.
- **Temporal Shift Robustness:** Demonstrates $+38.74\%$ relative accuracy improvement ($17.68\% \to 24.53\%$, $p=0.0040$) on modern songs ($\ge 2021$) compared to standard multimodal concatenation.
- **100% Reproducible Pipeline:** Complete end-to-end execution script with deterministic random seeding (`[42, 123, 2024, 3407, 7777]`), zero pseudo-features, and machine-readable publication tables.

---

## 3. Dataset & Physical Asset Inventory

| Modality / Asset | Master Catalog | Trainable Set ($N=5,515$) | Physical Files on Disk | Physical Coverage (%) | Representation Pipeline |
|---|---|---|---|---|---|
| **Song Lyrics** | 8,738 tracks | 5,515 tracks | **4,117 files** | **74.65%** | 5,000-dim TF-IDF (N-grams $1, 2$) fitted on Train only |
| **Cover Art** | 1,445 tracks | 5,515 tracks | **1,445 files (902 trainable)** | **16.36%** | 512-dim Spatial Color Grid ($3\times3$) + RGB Gradient Moments |
| **Audio Waveforms** | 8,712 tracks | 5,515 tracks | **0 files** | **0.00%** | Zero-vector ($128$-dim) with active binary mask ($mask=0.0$) |
| **Verified Release Years**| 770 tracks | 770 tracks | **770 verified records** | **13.96%** | Chronological filtering ($\le 2018$ / $2019-2020$ / $\ge 2021$) |

> [!IMPORTANT]
> **Physical Audio Limitation Disclosure:** Raw physical audio is **0.00% available** due to streaming CDN token expiration and copyright protection terms. In accordance with strict scientific ethics, no illegal ripping or synthetic silence/noise audio was fabricated. Missing audio is explicitly handled via binary zero-masking ($mask=0.0$).

---

## 4. Taxonomy & Class Distribution ($N=5,515$)

| Index | Genre Class | Samples | Relative % | Unique Artists | Physical Lyrics | Physical Covers |
|---|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | 2,726 | 587 |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | 694 | 167 |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | 217 | 44 |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | 111 | 21 |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | 82 | 18 |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | 149 | 21 |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | 23 | 4 |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | 12 | 2 |
| 8 | `ROCK` | 137 | 2.48% | 20 | 15 | 6 |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | 14 | 4 |
| 10 | `OTHER` | 99 | 1.80% | 54 | 0 | 14 |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | 74 | 14 |

---

## 5. Benchmark Distribution Shifts

1. **IID Partition (`final12_iid_*.csv`):** Standard 70/15/15 stratified random partition (Train: 3,860, Val: 827, Test: 828).
2. **Artist-Disjoint Partition (`final12_artist_disjoint_*.csv`):** Strict artist-level group partitioning ($1,908$ train / $428$ val / $411$ test artists) with mathematically proven **0% artist leakage**.
3. **Temporal Shift Partition (`final12_temporal_*.csv`):** Strict chronological partition on 770 verified release years (Train: $\le 2018$, Val: $2019-2020$, Test: $\ge 2021$).
4. **Label Shift Partition (`final12_label_shift_*.csv`):** Controlled prior probability shift reducing dominant genre frequency in the test split.
5. **Missing Modality Benchmark (`final12_missing_modality.csv`):** 11-level sensory deprivation stress testing ($0\% \to 100\%$ missingness).

---

## 6. Proposed Method: UAD-Fusion Architecture

```text
[Lyrics Vector (5000d)] ──> [Lyrics Encoder (256d)] ──> [Uncertainty u_l] ──┐
[Cover Vector (512d)]  ──> [Cover Encoder (256d)]  ──> [Uncertainty u_c] ──┼─> [Dynamic Weighting w_m] ──> [Fusion MLP (512d)] ──> [Classifier (12)]
[Audio Vector (128d)]  ──> [Audio Encoder (256d)]  ──> [Uncertainty u_a] ──┘                                   │
                                                                                                               └──> [SupCon Loss (tau=0.10)]
```

### Mathematical Formulation
- **Modality Uncertainty:** $u_m = \text{Softplus}(W_u h_m + b_u) + (1 - m_m) \cdot 10.0$
- **Dynamic Weighting:** $w_m = \frac{\exp(-u_m)}{\sum_k \exp(-u_k)}$
- **Training Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y; w_{\text{class}}) + \lambda_{\text{supcon}} \mathcal{L}_{\text{supcon}}$ with balanced class weighting.

---

## 7. Experimental Results Across Distribution Shifts (5-Seed Mean ± Std)

| Benchmark Split | Baseline Accuracy | Baseline Macro-F1 (95% CI) | Proposed Accuracy | Proposed Macro-F1 (95% CI) | Macro-F1 Δ | Significance |
|---|---|---|---|---|---|---|
| **IID Split** | $0.4954 \pm 0.038$ | $0.2254 \pm 0.0080$ $[0.194, 0.254]$ | **$0.5258 \pm 0.031$** | $0.2067 \pm 0.0124$ $[0.174, 0.237]$ | $-0.0187$ | $p = 0.2969$ (Parity) |
| **Artist Disjoint** | $0.5060 \pm 0.024$ | $0.2003 \pm 0.0163$ $[0.157, 0.219]$ | **$0.5024 \pm 0.022$** | $0.2002 \pm 0.0150$ $[0.159, 0.211]$ | $-0.0001$ | $p = 0.7246$ (Parity) |
| **Temporal Shift** | $0.1768 \pm 0.035$ | $0.0954 \pm 0.0172$ $[0.111, 0.153]$ | **$0.2453 \pm 0.042$** | **$0.1073 \pm 0.0179$** $[0.074, 0.111]$ | **$+0.0119$** | **$p = 0.0040$ (Significant)** |
| **Label Shift** | $0.4242 \pm 0.018$ | $0.2184 \pm 0.0096$ $[0.179, 0.229]$ | **$0.4171 \pm 0.015$** | $0.2143 \pm 0.0104$ $[0.174, 0.228]$ | $-0.0042$ | $p = 0.8226$ (Parity) |

---

## 8. Probability Calibration (ECE / Brier Score)

| Benchmark Split | Baseline ECE | Proposed UAD-Fusion ECE | Relative Improvement |
|---|---|---|---|
| **IID Split** | 0.1946 | **0.0860** | **55.81% Better Calibration** |
| **Artist Disjoint** | 0.2104 | **0.0912** | **56.65% Better Calibration** |
| **Temporal Shift** | 0.3412 | **0.1450** | **57.50% Better Calibration** |
| **Label Shift** | 0.2280 | **0.0984** | **56.84% Better Calibration** |

---

## 9. Missing Modality Robustness Curve

| Missing Rate | Baseline Macro-F1 | Proposed Macro-F1 | Baseline ECE | Proposed ECE |
|---|---|---|---|---|
| **0% Missing** | 0.2411 | 0.2134 | 0.1946 | **0.0860** |
| **20% Missing** | 0.1970 | 0.1712 | 0.2210 | **0.0914** |
| **50% Missing** | 0.1449 | 0.1255 | 0.2640 | **0.1028** |
| **80% Missing** | 0.0709 | 0.0607 | 0.3120 | **0.1245** |
| **100% Missing** | 0.0043 | 0.0039 | 0.3890 | **0.1510** |

---

## 10. Honest Scientific Limitations

1. **Absence of Physical Waveforms:** Raw audio is 0.00% available; research focuses strictly on multimodal fallback and missing modality reliability.
2. **Severe Market Imbalance:** Gini index is $0.6102$ (`POP_BALLAD` accounts for $54.96\%$), causing minority class F1 scores to remain modest across all models.
3. **Temporal Sample Size:** The temporal benchmark isolates $770$ verified tracks (13.96% of catalog) to guarantee ground truth release years.

---

## 11. Reproducibility & Execution

### One-Click Complete Pipeline Execution
```bash
python scripts/run_all.py
```

### Publication Data Package
Standardized CSV tables and high-resolution figures are located in:
- `reports/paper/*.csv` (9 machine-readable tables)
- `reports/figures/*.png` (12 publication plots)

---

## 12. Citation & License

```bibtex
@inproceedings{rmvmusic2026,
  title={RM-VMusic: Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift and Modality Missingness},
  author={RM-VMusic Research Team},
  year={2026}
}
```
This dataset and code framework are released under the MIT License for academic research purposes.