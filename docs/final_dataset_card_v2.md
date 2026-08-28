# RM-VMusic Final Dataset Card (v2 - 12-Class Release)

## 1. Overview
- **Dataset Name**: RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)
- **Version**: v1.0-RC1 (12-Class Multi-Split Release)
- **Master Samples**: **8,738 tracks**
- **Final Labeled Samples**: **5,514 tracks**
- **Number of Classes**: **12 classes** (11 target genres + `OTHER`)
- **OTHER Class Count**: **98 samples** (Religious sacred music, OST, Country)
- **Unique Artists**: **2,741 artists**
- **Verified Release Year Samples**: **770 tracks**

## 2. Physical Modality Status
- **Physical Audio Waveforms (`data/audio/`)**: **0 (0.00%)**
- **Physical Lyrics Files (`data/lyrics/`)**: **4,117 (74.66%)**
- **Physical Cover Images (`data/covers/`)**: **412 (7.47%)**
- **Full Multimodal Physical Samples**: **0 (0.00%)**
- **Dual Modality (Lyrics + Cover)**: **99 samples (1.80%)**
- **Lyrics Only Physical**: **4,018 samples (72.87%)**
- **Cover Only Physical**: **313 samples (5.68%)**
- **No Physical Files (Metadata Only)**: **1,084 samples (19.66%)**

## 3. 12-Class Distribution
| Genre | Samples ($N$) | Percentage (%) | Unique Artists |
|-------|---------------|----------------|----------------|
| `POP_BALLAD` | 3,031 | 54.97% | 1,888 |
| `BOLERO_TRUTINH` | 807 | 14.64% | 500 |
| `INSTRUMENTAL` | 287 | 5.20% | 141 |
| `RAP_HIPHOP` | 221 | 4.01% | 111 |
| `FOLK_TRADITIONAL` | 200 | 3.63% | 77 |
| `DANCE_EDM` | 193 | 3.50% | 139 |
| `REVOLUTIONARY` | 170 | 3.08% | 31 |
| `NHAC_TRINH` | 145 | 2.63% | 23 |
| `ROCK` | 137 | 2.48% | 20 |
| `RB_SOUL` | 132 | 2.39% | 27 |
| `OTHER` | 98 | 1.78% | 54 |
| `CHILDREN` | 93 | 1.69% | 41 |

## 4. Benchmark Splits (12 Classes)
1. `final_12class_iid_{train,val,test}.csv`: 3,859 / 827 / 828 (70 / 15 / 15 stratified).
2. `final_12class_artist_disjoint_{train,val,test}.csv`: 3,859 / 827 / 828 (Strict 0.00% artist leakage).
3. `final_12class_temporal_{train,val,test}.csv`: 526 / 54 / 190 (Chronologically verified).
4. `final_12class_label_shift_{train,val,test}.csv`: 3,904 / 799 / 811.
5. `final_12class_missing_modality.csv`: 5,514 annotated tracks.
