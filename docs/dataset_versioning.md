# RM-VMusic Dataset Versioning & Release Roadmap

This document formalizes the lifecycle stages and version definitions for the **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)** dataset.

---

## 1. Dataset Release Tiers

### RM-VMusic v0.x: Metadata & Harvesting Stage (Current Baseline)
- **Status**: Active Research Catalog
- **Master Tracks**: 8,738 | **Trainable Metadata Records**: 5,416 | **Tier C Queue**: 3,322
- **Audio**: 5,401 metadata stream URLs (historical tokens & entity links).
- **Lyrics**: 4,117 text annotations embedded in metadata table.
- **Covers**: 888 image URLs.
- **Physical Availability**: Lyrics materialized (`4,117 .txt`), Covers recovered (`413 .jpg`), Raw Audio pending physical download.
- **Purpose**: Establishes ground truth taxonomy, zero-leakage splits, duplicate audits, and metadata distribution shift benchmarks.

### RM-VMusic v1.0: Physically Verified & Materialized Multimodal Dataset (Target Phase 7)
- **Status**: Future Target
- **Criteria**:
  1. 100% of samples have verified physical waveform files on disk (`.wav` / `.mp3`, 16kHz/44.1kHz).
  2. 100% of vocal samples have aligned physical lyrics files (`.txt` / `.lrc`).
  3. Physical high-resolution cover art images (`.jpg` $\ge 300	imes300$).
  4. Complete checksum verification (SHA-256) and license provenance manifest.
