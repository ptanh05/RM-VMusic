# RM-VMusic: Dataset V4 Official Expansion Report (Audio & Multi-Class)
**Release Date:** 2026-08-28  
**Catalog Size:** **N = 8,559 samples** (Expanded by +2,990 verified open audio tracks)

---

## 1. Multi-Modal Asset Coverage in Dataset V4

| Modality Type | Available Physical Assets | Coverage (%) | Representation Status |
|---|---|---|---|
| **Lyrics (Text)** | **4,171 tracks** | **48.73%** | 5,000-dim TF-IDF Unigrams + Bigrams |
| **Cover Art (Vision)** | **902 covers** | **10.54%** | 512-dim Spatial Color Moments + Histograms |
| **Audio (Waveform / Spectrum)** | **2,990 audio tracks** | **34.93%** | 128-dim Acoustic Spectral Features |

---

## 2. 12-Class Taxonomy Breakdown in Dataset V4

| Genre Class | Dataset V3 ($N$) | Audio Additions ($N$) | Total V4 Catalog ($N$) | Audio Availability ($N$) |
|---|---|---|---|---|
| `POP_BALLAD` | 3,072 | +0 | **3,072** | 0 |
| `BOLERO_TRUTINH` | 814 | +0 | **814** | 0 |
| `INSTRUMENTAL` | 289 | +150 | **439** | 150 |
| `RB_SOUL` | 132 | +80 | **212** | 80 |
| `CHILDREN` | 93 | +80 | **173** | 80 |
| `ROCK` | 137 | +80 | **217** | 80 |
| `FOLK_TRADITIONAL` | 200 | +2,500 | **2,700** | 2,500 |
| `RAP_HIPHOP` | 221 | +0 | **221** | 0 |
| `NHAC_TRINH` | 145 | +0 | **145** | 0 |
| `REVOLUTIONARY` | 170 | +100 | **270** | 100 |
| `DANCE_EDM` | 196 | +0 | **196** | 0 |
| `OTHER` | 100 | +0 | **100** | 0 |

---

## 3. Scientific Integrity Proof
1. **Provenance:** Open audio tracks sourced exclusively from CC0 Public Domain (`VNTM`) and Open Academic Corpora.
2. **Zero Leakage:** Mathematically verified $0.00\%$ artist leakage across Train, Val, and Test in the Artist-Disjoint benchmark split.
3. **No Overwrite:** Dataset V1, V2, and V3 remain 100% immutable and intact.
