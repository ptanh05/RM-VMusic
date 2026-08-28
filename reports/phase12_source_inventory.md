# RM-VMusic Phase 12: Comprehensive Internal & External Source Inventory
**Evaluation Date:** 2026-08-28

---

## 1. Internal Workspace Data Inventory

| Source Identifier | Path | Total Records | Modality Content | License / Terms | Usability Tier |
|---|---|---|---|---|---|
| `sunbv56/sunbv56_train_full.jsonl` | `data\raw\sunbv56_train_full.jsonl` | 7,201 | Metadata + Timestamps + Lyrics | Open Academic Research (Hugging Face) | **HIGH (Lyrics & Metadata)** |
| `sunbv56/sunbv56_pilot_train.jsonl` | `data\raw\sunbv56_pilot_train.jsonl` | 1,500 | Metadata + Timestamps + Lyrics | Open Academic Research (Hugging Face) | **HIGH (Lyrics & Metadata)** |
| `sunbv56/sunbv56_eval.jsonl` | `data\raw\sunbv56_eval.jsonl` | 643 | Metadata + Timestamps + Lyrics | Open Academic Research (Hugging Face) | **HIGH (Lyrics & Metadata)** |
| `VietLyrics/vietlyrics_train_7k.csv` | `data\raw\vietlyrics_train_7k.csv` | 7,433 | Metadata + Lyrics + Genre Tags | CC-BY-NC-SA 4.0 / Open Research | **VERY HIGH (Core Genre Ground Truth)** |
| `VietLyrics/vietlyrics_val_1k.csv` | `data\raw\vietlyrics_val_1k.csv` | 995 | Metadata + Lyrics + Genre Tags | CC-BY-NC-SA 4.0 / Open Research | **VERY HIGH (Core Genre Ground Truth)** |
| `RM-VMusic Master Catalog` | `data\processed\master_metadata.csv` | 8,738 | Metadata + Physical Links + Release Years | MIT / Academic | **CORE MASTER BASELINE** |

---

## 2. Key Discoveries
1. **VietLyrics (`tsdocode/vi-song-7k-public` / `BatmanofZuhandArrgh/VietLyrics`):** Contains 8,428 total tracks across train and val, providing explicit genre labels for 4,969 tracks under open academic terms.
2. **sunbv56 (`sunbv56/song_dataset`):** Contains 9,344 total JSONL records with complete word-level timestamp alignments.
3. **Master Catalog (`master_metadata.csv`):** 8,738 tracks representing the consolidated catalog uniting Zing MP3 links, VietLyrics annotations, and physical local files.
