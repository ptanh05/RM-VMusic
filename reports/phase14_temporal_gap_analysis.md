# RM-VMusic Phase 14: Temporal Field Taxonomy & Gap Report
**Evaluation Date:** 2026-08-28

---

## 1. Strict Separation of Temporal Fields

| Temporal Concept | Definition in RM-VMusic | Benchmark Eligibility | Evidence Requirement |
|---|---|---|---|
| **`composition_year`** | The historical year the composer wrote the musical score/lyrics (e.g., 1970 for Trịnh Công Sơn) | **NOT ELIGIBLE FOR DISTRIBUTION SHIFT** | Musicological biography only |
| **`recording_year`** | The date the studio session or master tape was recorded | **SECONDARY CONTEXT** | Studio recording log |
| **`album_year`** | The publication year of the compilation/album | **PROXIMATE METADATA** | Physical CD/cassette booklet |
| **`release_year`** | The verified commercial publication/digital release date of the specific audio track | **CORE TEMPORAL BENCHMARK FIELD** | Official digital publication tag |

---

## 2. Release Year Distribution Summary ($N = 5,569$)

- **Verified Release Year:** **770 tracks (13.83%)**
- **Unverified / Missing Year:** **4,799 tracks (86.17%)**
- **Partition Distribution:**
  - Historical Train ($\le 2018$): **526 tracks**
  - Transition Val ($2019–2020$): **54 tracks**
  - Modern Test ($\ge 2021$): **190 tracks**
- **Active Test Classes:** **10 / 12 classes** (`POP_BALLAD`, `BOLERO_TRUTINH`, `INSTRUMENTAL`, `RAP_HIPHOP`, `FOLK_TRADITIONAL`, `DANCE_EDM`, `REVOLUTIONARY`, `ROCK`, `RB_SOUL`, `OTHER`).
- **Data Availability Limitation:** `NHAC_TRINH` ($N=0$) and `CHILDREN` ($N=0$) remain authentically absent in post-2021 open releases.
