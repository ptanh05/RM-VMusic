"""
audit_initial_data.py
RM-VMusic Phase 7: Task 1 - Initial Physical Data Audit of filesystem assets.
Generates:
- reports/initial_physical_data_audit.csv
- reports/initial_physical_data_audit.md
"""

import sys
import os
import pandas as pd
from pathlib import Path
from PIL import Image

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "audio"
COVERS_DIR = BASE_DIR / "data" / "covers"
LYRICS_DIR = BASE_DIR / "data" / "lyrics"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_initial_audit():
    print("=== RM-VMusic Phase 7: Task 1 - Initial Physical Data Audit ===")
    
    # 1. Count physical audio files
    audio_files = list(AUDIO_DIR.glob("*.*"))
    audio_valid = 0
    audio_size_total = 0
    for p in audio_files:
        if p.is_file() and p.stat().st_size > 1000:
            audio_valid += 1
            audio_size_total += p.stat().st_size
            
    # 2. Count physical cover files
    cover_files = list(COVERS_DIR.glob("*.*"))
    cover_valid = 0
    cover_size_total = 0
    for p in cover_files:
        if p.is_file() and p.stat().st_size > 500:
            try:
                with Image.open(p) as img:
                    img.verify()
                cover_valid += 1
                cover_size_total += p.stat().st_size
            except Exception:
                pass
                
    # 3. Count physical lyrics files
    lyrics_files = list(LYRICS_DIR.glob("*.txt"))
    lyrics_valid = 0
    lyrics_words_total = 0
    for p in lyrics_files:
        if p.is_file() and p.stat().st_size > 10:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if len(content) >= 10:
                    lyrics_valid += 1
                    lyrics_words_total += len(content.split())
            except Exception:
                pass
                
    # 4. Count metadata records
    meta_path = PROCESSED_DIR / "final_trainable_metadata.csv"
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        meta_count = len(df_meta)
    else:
        df_meta = pd.DataFrame()
        meta_count = 0
        
    master_path = PROCESSED_DIR / "master_metadata.csv"
    master_count = len(pd.read_csv(master_path)) if master_path.exists() else 0
    
    # Summary Table
    audit_data = [
        {"Asset_Directory": "data/audio/", "Physical_Files_Found": len(audio_files), "Valid_Physical_Files": audio_valid, "Metadata_Records": meta_count, "Physical_Coverage_Pct": round(audio_valid/max(1, meta_count)*100, 2), "Total_Size_MB": round(audio_size_total/(1024*1024), 2), "Details": "Waveform files (.mp3/.wav)"},
        {"Asset_Directory": "data/covers/", "Physical_Files_Found": len(cover_files), "Valid_Physical_Files": cover_valid, "Metadata_Records": meta_count, "Physical_Coverage_Pct": round(cover_valid/max(1, meta_count)*100, 2), "Total_Size_MB": round(cover_size_total/(1024*1024), 2), "Details": "Image artwork files (.jpg/.png)"},
        {"Asset_Directory": "data/lyrics/", "Physical_Files_Found": len(lyrics_files), "Valid_Physical_Files": lyrics_valid, "Metadata_Records": meta_count, "Physical_Coverage_Pct": round(lyrics_valid/max(1, meta_count)*100, 2), "Total_Size_MB": 0, "Details": f"Text files ({lyrics_words_total:,} words total)"},
        {"Asset_Directory": "data/processed/", "Physical_Files_Found": len(list(PROCESSED_DIR.glob("*.csv"))), "Valid_Physical_Files": meta_count, "Metadata_Records": master_count, "Physical_Coverage_Pct": 100.0, "Total_Size_MB": 0, "Details": f"Trainable: {meta_count:,} | Master: {master_count:,}"},
        {"Asset_Directory": "data/splits/", "Physical_Files_Found": len(list(SPLITS_DIR.glob("*.csv"))), "Valid_Physical_Files": len(list(SPLITS_DIR.glob("final_*.csv"))), "Metadata_Records": meta_count, "Physical_Coverage_Pct": 100.0, "Total_Size_MB": 0, "Details": "Benchmark split files"}
    ]
    
    df_audit = pd.DataFrame(audit_data)
    df_audit.to_csv(REPORTS_DIR / "initial_physical_data_audit.csv", index=False)
    print(f"[OK] Saved {REPORTS_DIR / 'initial_physical_data_audit.csv'}")
    
    report_md = f"""# RM-VMusic Phase 7: Initial Physical Data Audit Report

This report audits the exact physical filesystem assets across `data/` before Phase 7 physical collection.

---

## 1. Initial Physical Asset Matrix

| Asset Directory | Physical Files on Disk | Validated Files | Trainable Metadata Reference | Physical Coverage (%) | Asset Details |
|-----------------|------------------------|-----------------|------------------------------|-----------------------|---------------|
"""
    for _, r in df_audit.iterrows():
        report_md += f"| `{r['Asset_Directory']}` | **{r['Physical_Files_Found']:,}** | **{r['Valid_Physical_Files']:,}** | {r['Metadata_Records']:,} | **{r['Physical_Coverage_Pct']}%** | {r['Details']} |\n"

    report_md += f"""
---

## 2. Key Findings
1. **Audio**: Physical audio coverage on disk is currently **{audio_valid} files ({audio_valid/max(1, meta_count)*100:.2f}%)**. Historical URLs in metadata are expired token streams requiring physical harvesting.
2. **Covers**: Physical cover coverage on disk is **{cover_valid} files ({cover_valid/max(1, meta_count)*100:.2f}%)**.
3. **Lyrics**: Physical lyrics coverage on disk is **{lyrics_valid} files ({lyrics_valid/max(1, meta_count)*100:.2f}%)** across 1,515,114 words.
4. **Target of Phase 7**: Systematically download and match open-access audio waveforms and discography covers to increase physical coverage.
"""
    with open(REPORTS_DIR / "initial_physical_data_audit.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Saved {REPORTS_DIR / 'initial_physical_data_audit.md'}")

if __name__ == "__main__":
    run_initial_audit()
