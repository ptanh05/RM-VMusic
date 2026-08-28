# RM-VMusic Phase 16: Provenance & Anti-Derivative Dataset Audit
**Evaluation Date:** 2026-08-28

---

## 1. Multi-Source Lineage & Derivative Classification

| Dataset Artifact | Canonical Publisher | Dataset Heritage | Derivative Status | Final Ingestion Verdict |
|---|---|---|---|---|
| `VietLyrics (`tsdocode/vi-song-7k-public`)` | VietLyrics Research Group (arXiv:2403.07823) | Original academic data collection with human-curated alignment | **`CANONICAL_PRIMARY_SOURCE`** | 100% Ingested in V1/V2/V3 ($N=5,569$) |
| `sunbv56 / song_dataset` | sunbv56 (Hugging Face) | Phonetic word-level timestamp alignment corpus | **`CANONICAL_PRIMARY_SOURCE`** | 100% Ingested in V1/V2/V3 ($N=5,569$) |
| `Vietnam Traditional Music (VNTM / LTPhat)` | LTPhat (Kaggle / GitHub) | Original traditional acoustic audio collection (5 genres) | **`CANONICAL_PRIMARY_SOURCE`** | Audited traditional acoustic reference benchmark |
| `NTQAI / Vietnamese-Traditional-Music` | NTQ Solution AI Lab | Audio clips for traditional music classification | **`INDEPENDENT_OPEN_SOURCE`** | Audio-only dataset lacking text lyrics and release years |
| `Whisper Vietnamese Lyrics / kelvinbksoh` | kelvinbksoh | Direct fork and repackaging of sunbv56 dataset | **`DERIVATIVE_MIRROR (Rejected)`** | REJECTED_ALREADY_USED (Contains 0 unique samples) |

---

## 2. Anti-Derivative Certification
Every external repository evaluated during Phase 16 was cross-examined against primary canonical sources. All derivative mirrors and secondary forks were formally identified and rejected to prevent duplicate sample contamination.
