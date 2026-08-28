# RM-VMusic Phase 15: Multi-Source Provenance & Anti-Mirror Audit
**Evaluation Date:** 2026-08-28

---

## 1. Lineage & Provenance Certification

| Repository Artifact | Original Publisher | Collection Methodology | Ground Truth Grounding | Final Lineage Determination |
|---|---|---|---|---|
| `VietLyrics (`tsdocode/vi-song-7k-public`)` | VietLyrics Research Group (arXiv:2403.07823) | Standardized academic crawl & manual lyric alignment | Explicit multi-genre taxonomy tags from curated catalog | **`PRIMARY_CANONICAL_SOURCE (Tier 1)`** |
| `sunbv56 / song_dataset` | sunbv56 (Hugging Face) | Word-level timestamped lyric alignment dataset | Phonetic time-aligned lyrics corpus | **`PRIMARY_CANONICAL_SOURCE (Tier 1)`** |
| `Vietnam Traditional Music (VNTM)` | LTPhat / Kaggle Research Community | Mel-Spectrogram audio clips of traditional Vietnamese music | 5 traditional genres (Ca trù, Chèo, Chầu văn, Hát xẩm, Dân ca) | **`PRIMARY_CANONICAL_SOURCE (Tier 1)`** |
| `Downstream GitHub Scrapers / Zalo AI Forks` | Individual GitHub Users / Student repositories | Secondary scrape / direct repackaging of sunbv56 / Zing MP3 | Uncurated / mirror duplicates | **`REJECTED_ALREADY_USED_OR_FORK (Tier 3)`** |

---

## 2. Scientific Anti-Mirror Rules
1. **Zero Repackaged Inflations:** Any dataset identified as a mirror, fork, or direct derivative of `VietLyrics`, `sunbv56`, or `VNTM` is explicitly flagged and excluded from new candidate counts.
2. **Deterministic Provenance Tracking:** Every sample in the master dataset (`final_12class_metadata_v3.csv`) contains exact `source`, `source_id`, and `label_source` columns tracing back to primary academic publications.
