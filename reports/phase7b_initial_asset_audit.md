# RM-VMusic Phase 7B: Initial Asset Audit & Feasibility Assessment
**Audit Date:** 2026-08-28  
**Scope:** Master Catalog ($N=8,738$) & Trainable Benchmark Dataset ($N=5,416$)  
**Status:** Baseline Physical Audit Complete

---

## 1. Audit Overview & Physical Reality

| Modality | Metadata Claims Available | Direct Stream/URL Indexed | Physical Files on Local Disk | Valid Physical Files | Real Physical Coverage |
|---|---|---|---|---|---|
| **Audio** | 5,416 tracks | 5,416 tracks | **0 files** | **0 files** | **0.00%** |
| **Album Covers** | 413 tracks | 413 tracks | **412 files** | **412 files** | **7.61%** |
| **Lyrics** | 4,117 tracks | 4,117 tracks | **4,117 files** | **4,117 files** | **76.02%** |
| **Release Year** | 768 tracks | N/A | 768 tracks | 768 tracks | **14.18%** |

---

## 2. Detailed Modality Audit

### A. Audio Waveforms (`data/audio/`)
- **Physical Files:** 0 files in `data/audio/`.
- **Remote Endpoints in Metadata:** 5,416 tracks have `source_url` pointing to Zing MP3 CDN stream cache (`a128-z3.zmdcdn.me`).
- **Legal & Technical Constraint:** 
  1. The CDN links carry temporary HMAC authentication tokens (e.g. `exp=1772900208~acl=...`) generated during raw crawl caching.
  2. Downloading copyrighted streaming tracks at scale without individual track-level redistribution rights violates copyright guidelines.
  3. Under Phase 7B rules, copyrighted streaming music must **never** be downloaded via bypass/scraping mechanisms (yt-dlp, stream-ripping), and no synthetic/silence/noise audio may be fabricated.
  4. Records lacking verifiable public domain/creative commons or open research waveforms are marked strictly as `audio_status = unavailable`.

### B. Album Covers (`data/covers/`)
- **Physical Files on Disk:** 412 `.jpg` images (17.65 MB).
- **Integrity Validation:** 100% (412/412) are valid, uncorrupted JPEG files with non-zero dimensions (typically $240 \times 240$ px).
- **Missing Covers:** 5,004 tracks have no physical cover image on disk.
- **Action for Phase 7B:** Run legitimate cover materialization for public promotional artwork URLs that remain accessible and unexpired.

### C. Lyrics (`data/lyrics/`)
- **Physical Files on Disk:** 4,117 `.txt` files (8.23 MB).
- **Integrity Validation:** 
  - Encoding: 100% UTF-8 (NFC normalized Vietnamese text).
  - Empty files: 0 files ($0.00\%$).
  - Average word count: 284 words per song.
  - Coverage: 76.02% of trainable tracks ($4,117 / 5,416$).
  - Missing lyrics: 1,299 tracks ($23.98\%$) have no text lyrics (primarily `INSTRUMENTAL` tracks and tracks without published lyrics).

---

## 3. Metadata & Column Schema Audit

The primary canonical file `data/processed/final_trainable_metadata.csv` contains 22 fields:
- `song_id`: Unique track identifier (`RMVM_S_<hash>`).
- `title`, `artist`, `artist_id`: Standardized metadata (0 duplicates across entire catalog).
- `genre`, `raw_genre`, `source_genre`: Standardized 11-class genre mapping.
- `tier`: Quality tier (`TIER_A` = 4,157, `TIER_B` = 1,259).
- `label_source`, `label_confidence`: Provenance documentation.
- `release_year`: Verified release year (768 tracks).
- `audio_path`, `lyrics_path`, `cover_path`: Target local asset paths.
- `source_url`, `source_id`: Upstream source index.

---

## 4. Immediate Phase 7B Next Actions

1. **Cover Materialization:** Attempt downloading remaining reachable cover images into `data/covers/` using `scripts/materialize_covers.py` with strict validation.
2. **Audio Materialization Pipeline:** Build `scripts/materialize_audio.py` to audit open audio access and produce `data/processed/audio_manifest.csv`.
3. **Lyrics Manifestation:** Produce `data/processed/lyrics_manifest.csv`.
4. **Eliminate Pseudo-Features:** Ban SHA-256 hash embeddings from baseline and proposed pipelines. Modality absence must be represented by true zero-masks.
5. **Evaluate OTHER Class:** Rigorously analyze candidate records for an explicit 12th class (`OTHER`).
6. **Re-split Dataset:** Rebuild checksummed stratified splits on final dataset.
7. **Re-run Baseline Experiments:** Benchmark on physical features (real lyrics TF-IDF + real cover visual features + missing modality masking).
