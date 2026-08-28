# RM-VMusic Final Dataset Card

## 1. Dataset Description
- **Dataset Name**: RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)
- **Version**: v0.9 (Pre-Physical Audio Benchmark Release)
- **Dataset Size**: **5,416 trainable tracks** | **8,738 master catalog tracks**
- **Number of Classes**: **11 target genres**
- **Unique Artists**: **2,707 artists**

## 2. Modality Physical Availability
- **Physical Audio Files (`data/audio/`)**: **0 (0.00%)**
- **Physical Lyrics Files (`data/lyrics/`)**: **4,117 (76.02%)**
- **Physical Cover Images (`data/covers/`)**: **413 (7.63%)**
- **Full Multimodal Samples (Audio+Lyrics+Cover)**: **0 (0.00%)**
- **Verified Release Year Samples**: **768 (14.18%)**

## 3. Class Distribution
| Genre Code | Sample Count ($N$) | Percentage (%) |
|------------|--------------------|----------------|
| `POP_BALLAD` | 3,031 | 55.96% |
| `BOLERO_TRUTINH` | 807 | 14.90% |
| `INSTRUMENTAL` | 287 | 5.30% |
| `FOLK_TRADITIONAL` | 159 | 2.94% |
| `DANCE_EDM` | 154 | 2.84% |
| `RAP_HIPHOP` | 152 | 2.81% |
| `REVOLUTIONARY` | 95 | 1.75% |
| `CHILDREN` | 85 | 1.57% |
| `ROCK` | 83 | 1.53% |
| `NHAC_TRINH` | 78 | 1.44% |
| `RB_SOUL` | 69 | 1.27% |

## 4. Benchmark Splits Provided
1. `final_iid_{train,val,test}.csv`: 70% / 15% / 15% stratified.
2. `final_artist_disjoint_{train,val,test}.csv`: 70% / 15% / 15% with strict 0.00% artist overlap.
3. `final_temporal_{train,val,test}.csv`: $\le 2018$ (526), $2019-2020$ (54), $\ge 2021$ (188).
4. `final_label_shift_{train,val,test}.csv`: Controlled distribution shift.
5. `final_missing_modality.csv`: Physical modality pattern annotations.

## 5. Known Limitations & Recommended Usage
- **Recommended Usage**: Vietnamese music genre classification using Lyrics NLP, text-cover multimodal fusion, distribution shift benchmarking, and class imbalance research.
- **Not Recommended Usage**: End-to-end raw waveform audio modeling without prior physical audio downloading.
