# RM-VMusic Phase 7B: Complete Technical & Empirical Methodology Specification
**Document Version:** 2.0 (Phase 7B Release)  
**Author:** RM-VMusic Research Team  
**Date:** 2026-08-28

---

## 1. Overview & Research Paradigm

The **RM-VMusic** benchmark investigates **Multimodal Vietnamese Music Genre Classification under Real-World Distribution Shift**. In real streaming ecosystems, metadata, album covers, and lyrics are frequently incomplete, while raw audio streams may be restricted by licensing or bandwidth constraints.

Phase 7B establishes an uncompromising, reproducible foundation by:
1. Materializing and validating all physical assets on disk (`data/lyrics/`, `data/covers/`, `data/audio/`).
2. Completely eliminating pseudo-hash and random features from the modeling pipeline.
3. Structuring an explicit 12-class taxonomy with a semantically verified `OTHER` class.
4. Constructing 5 distribution-shift benchmark partitions with strict 0% artist leakage.
5. Re-executing all baseline models on true physical multimodal representations.

---

## 2. Taxonomy & Semantic Construction of `OTHER`

The standardized 12-class Vietnamese music genre taxonomy is formally specified as follows:

| Class ID | Genre Identifier | Semantic Scope & Target Music Sub-genres | Source Sample Count |
|---|---|---|---|
| 0 | `POP_BALLAD` | Contemporary Vietnamese pop, lyrical ballads, acoustic slow-tempo songs | 3,031 (54.96%) |
| 1 | `BOLERO_TRUTINH` | Classic Southern Vietnamese Bolero, nostalgic love ballads, Rhumba/Chacha | 807 (14.63%) |
| 2 | `INSTRUMENTAL` | Instrumental solos (guitar, piano, saxophone, traditional instruments) | 287 (5.20%) |
| 3 | `RAP_HIPHOP` | Vietnamese hip-hop, underground rap, boom-bap, trap | 221 (4.01%) |
| 4 | `FOLK_TRADITIONAL` | Regional folk music, Quan họ, Ca trù, Vọng cổ, Chèo, ethnic minority melodies | 200 (3.63%) |
| 5 | `DANCE_EDM` | Electronic dance music, electro-house, club remixes, Vinahouse | 193 (3.50%) |
| 6 | `REVOLUTIONARY` | Patriotic Red music (*Nhạc Đỏ*), heroic military anthems, historical songs | 170 (3.08%) |
| 7 | `NHAC_TRINH` | Philosophical and acoustic compositions by Trịnh Công Sơn | 145 (2.63%) |
| 8 | `ROCK` | Vietnamese alternative rock, heavy metal, hard rock, punk rock | 137 (2.48%) |
| 9 | `RB_SOUL` | Contemporary Vietnamese R&B, neo-soul, groove, slow jam | 132 (2.39%) |
| 10 | `OTHER` | Verified out-of-taxonomy genres (Religious/Sacred hymns, Film OSTs, Country) | 99 (1.80%) |
| 11 | `CHILDREN` | Nursery rhymes, elementary school songs, lullabies | 93 (1.69%) |

---

## 3. Physical Asset Verification & Manifest Protocols

### A. Lyrics Modality (`data/lyrics/`)
- **Format:** Text files (`<song_id>.txt`) decoded with Unicode NFC normalization.
- **Audit Tool:** `scripts/materialize_lyrics.py`.
- **Integrity Rule:** Tracks with 0-byte or corrupted text files are excluded (`mask = 0.0`).
- **Feature Extraction:** 5,000-dimensional TF-IDF (unigram + bigram) fitted strictly on the training partition.

### B. Cover Art Modality (`data/covers/`)
- **Format:** JPEG images (`<song_id>.jpg`) decoded with Pillow.
- **Audit Tool:** `scripts/materialize_covers.py` (16 concurrent workers with timeout=5s).
- **Feature Extraction:** 512-dimensional spatial color histograms ($3 \times 3$ grid $\times 16$ bins/channel = 432 dims) combined with global RGB moments and gradient texture descriptors (80 dims).

### C. Audio Modality (`data/audio/`)
- **Legal Compliance:** In accordance with open research ethics, commercial streams with expiring HMAC tokens are not scraped.
- **Representation:** Missing audio waveforms are explicitly assigned zero-vectors ($128$-dim) with active mask $m_{\text{audio}} = 0.0$.

---

## 4. Benchmark Split Construction (5 Distribution Shifts)

1. **IID Partition (`final12_iid_*.csv`):** Standard 70/15/15 stratified random split by genre class.
2. **Artist-Disjoint Partition (`final12_artist_disjoint_*.csv`):** Strict artist-level group partitioning. Proven $0\%$ leakage:
   $$\text{Train Artists} \cap \text{Val Artists} = \emptyset, \quad \text{Train Artists} \cap \text{Test Artists} = \emptyset, \quad \text{Val Artists} \cap \text{Test Artists} = \emptyset$$
3. **Temporal Partition (`final12_temporal_*.csv`):** Chronological split on verified release years (Train: $\le 2018$, Val: $2019-2020$, Test: $\ge 2021$).
4. **Label Shift Partition (`final12_label_shift_*.csv`):** Controlled prior shift reducing majority class prevalence and amplifying minority class representation in the test set.
5. **Missing Modality Partition (`final12_missing_modality.csv`):** Dynamic modality ablation evaluating system robustness under sensory deprivation.

---

## 5. Modeling & Optimization Standards

- **Loss Function:** Balanced Cross-Entropy Loss:
  $$\mathcal{L} = -\sum_{c=1}^{C} w_c \cdot y_c \log \hat{y}_c, \quad w_c = \frac{N}{C \cdot N_c} \quad (\text{computed on train split})$$
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Early Stopping:** Patience of 8 epochs based on Validation Macro-$F_1$.
- **Evaluation Metrics:** Accuracy, Macro-$F_1$, Weighted-$F_1$, Balanced Accuracy, Per-Class Precision/Recall/$F_1$, Confusion Matrix.
