# RM-VMusic: Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift

A research benchmark and dataset pipeline for studying **Distribution Shift** and **Multimodal Reliability** (Audio + Lyrics + Metadata) in Vietnamese Music Genre Classification.

---

## 📁 Repository Structure

```text
RM-VMusic/
├── README.md
├── requirements.txt
│
├── docs/
│   ├── dataset_audit.md        # Technical audit of all examined data sources
│   └── genre_taxonomy.md       # Standardized Vietnamese genre taxonomy & mapping rules
│
├── data/
│   ├── raw/                    # Cached raw source downloads
│   ├── processed/
│   │   ├── master_metadata.csv # Master dataset conforming to 18-field schema
│   │   └── rejected_records.csv# Fully tracked rejected records with reasons
│   ├── audio/                  # Audio features / cached data (gitignored)
│   ├── lyrics/                 # Preprocessed lyrics
│   ├── covers/                 # Album cover art
│   └── splits/                 # Benchmark distribution shift splits
│       ├── iid.csv             # Standard Stratified IID Split
│       ├── artist_disjoint.csv # Strict 0% Artist Leakage Split
│       ├── temporal.csv        # Temporal Shift Split
│       ├── missing_modality.csv# Robustness under missing audio/lyrics
│       └── label_shift.csv     # Prior Label Probability Shift
│
├── scripts/
│   ├── audit_sources.py        # Audits live data sources & outputs dataset_audit.md
│   ├── build_dataset.py        # Cleans, normalizes, & curates master pilot dataset
│   ├── check_duplicates.py     # Checks duplicates across titles, artists, IDs, and hashes
│   ├── check_artist_leakage.py # Validates 0% artist leakage on disjoint splits
│   ├── create_splits.py        # Generates all 5 distribution shift splits
│   └── generate_report.py      # Produces dataset_quality_report.md
│
└── reports/
    └── dataset_quality_report.md # Comprehensive data quality & modality completeness report
```

---

## 📋 Master Schema (18 Canonical Fields)

| Field Name | Type | Description |
|------------|------|-------------|
| `song_id` | String | Unique canonical song identifier (`RMVM_S<hash>`) |
| `title` | String | Cleaned, normalized song title (Unicode NFC) |
| `artist` | String | Standardized artist name(s) |
| `artist_id` | String | Unique deterministic artist ID (`ART_<hash>`) |
| `album` | String | Album name or `NULL` |
| `album_id` | String | Unique deterministic album ID (`ALB_<hash>`) or `NULL` |
| `genre` | String | Normalized genre class or `NEEDS_MANUAL_ANNOTATION` |
| `release_year` | Integer | Release year or `NULL` |
| `audio_path` | String | Local cached audio path or `NULL` |
| `audio_url` | String | Remote streaming/reference URL |
| `lyrics` | String | Full Vietnamese lyrics text (newlines preserved) or `NULL` |
| `cover_path` | String | Local cover image path or `NULL` |
| `cover_url` | String | Remote cover image URL or `NULL` |
| `source` | String | Origin dataset identifier |
| `source_id` | String | Original ID in source dataset (e.g. Zing MP3 ID) |
| `annotation_status`| String | `cross_verified`, `normalized`, or `needs_annotation` |
| `annotator_id` | String | System annotator tag (`pipeline_v1`) |
| `annotation_agreement` | Float | Confidence / agreement score or `NULL` |

---

## 🚀 Pipeline Execution Workflow

To run the complete pipeline from scratch:

```bash
# 1. Audit remote sources
python scripts/audit_sources.py

# 2. Build cleaned master dataset & pilot
python scripts/build_dataset.py

# 3. Analyze duplicates
python scripts/check_duplicates.py

# 4. Generate all 5 distribution shift splits
python scripts/create_splits.py

# 5. Verify artist leakage
python scripts/check_artist_leakage.py

# 6. Generate comprehensive quality report
python scripts/generate_report.py
```

---

## ⚖️ Legal & Ethical Compliance
- Raw copyrighted audio files are **never** committed to the repository.
- Only reference URLs, streaming endpoints, and processed metadata are indexed.
- Adheres to academic fair-use guidelines for machine learning research.