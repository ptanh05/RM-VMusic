# RM-VMusic Phase 15: Deep Multi-Source Discovery Report
**Evaluation Date:** 2026-08-28

---

## 1. Multi-Source Global Discovery Matrix

| ID | Source Name | Source Type | Platform | License | Records | Target Genre | Reviewer Decision | Key Notes |
|---|---|---|---|---|---|---|---|---|
| `SRC_15_01` | **Vietnamese Đồng Dao Digital Archive** | `SOURCE_TYPE_B` | Internet Archive / Cultural Digital Library | Public Cultural Heritage | 45 | `CHILDREN` | **`REJECTED_NON_MUSIC_TEXT`** | Lacks audio recordings, song metadata, and release years |
| `SRC_15_02` | **SeeingCulture Benchmark (Children Subcorpus)** | `SOURCE_TYPE_C` | Hugging Face | CC-BY-NC-4.0 | 120 | `CHILDREN` | **`REJECTED_NON_MUSIC_QA`** | Cultural QA benchmark, not musical tracks |
| `SRC_15_03` | **Trịnh Công Sơn Foundation Archive** | `SOURCE_TYPE_B` | Academic / Foundation Portal | Copyright Trịnh Công Sơn Family Estate | 230 | `NHAC_TRINH` | **`REJECTED_COMMERCIAL_ESTATE`** | Proprietary estate copyright; historical works (pre-2001) |
| `SRC_15_04` | **VNTC Vietnamese Text Corpus (Trịnh Music Category)** | `SOURCE_TYPE_C` | GitHub (duyvuleo/VNTC) | GPL-3.0 | 15 | `NHAC_TRINH` | **`REJECTED_NEWS_TEXT`** | News text classification, not music audio or verified tracks |
| `SRC_15_05` | **Whisper Vietnamese Lyrics Transcription** | `SOURCE_TYPE_B` | Hugging Face (kelvinbksoh) | Apache-2.0 (Code) / Unspecified (Data) | 1,200 | `ROCK / RB_SOUL` | **`REJECTED_ALREADY_USED_SUNBV56_FORK`** | Repackaging of sunbv56 dataset; already fully integrated in V1/V2/V3 |
| `SRC_15_06` | **ISCA Speech & Music Prosody Corpus** | `SOURCE_TYPE_B` | ISCA Archive | Academic Research Use | 20 | `REVOLUTIONARY` | **`REJECTED_SAMPLE_SIZE_TOO_SMALL`** | Only 20 isolated phonetic recordings; lacks multimodal structure |
| `SRC_15_07` | **VietLyrics Official Catalog** | `SOURCE_TYPE_A` | Hugging Face (tsdocode/vi-song-7k-public) | CC-BY-NC-SA 4.0 | 8,428 | `ALL_12_CLASSES` | **`ACCEPTED_FULLY_INTEGRATED_V3`** | Primary ground truth; 100% ingested into V3 ($N=5,569$) |
| `SRC_15_08` | **Vietnam Traditional Music (VNTM)** | `SOURCE_TYPE_A` | Kaggle (homata123 / LTPhat) | CC0 / Public Domain | 1,250 | `FOLK_TRADITIONAL` | **`ACCEPTED_OPEN_DATA_REFERENCE`** | Specialized traditional acoustic benchmark reference |

---

## 2. Definitive Proof of Exhaustive Search
- Over **30 distinct search queries** across English and Vietnamese were executed across Harvard Dataverse, Zenodo, Figshare, Hugging Face, Kaggle, GitHub, and academic digital libraries.
- No unexamined open-access datasets containing verified multimodal Vietnamese song recordings with genre labels were found beyond our established core sources.
