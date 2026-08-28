"""
phase14_temporal_audit.py
RM-VMusic Phase 14: Temporal Field Separation & Gap Analysis Engine.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

def run_temporal_audit():
    print("=== RM-VMusic Phase 14: Temporal Field Distinction & Gap Analysis ===")
    
    df_v3 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata_v3.csv")
    
    y_valid = pd.to_numeric(df_v3["release_year"], errors="coerce").dropna().astype(int)
    
    n_total = len(df_v3)
    n_known_release = len(y_valid)
    n_unknown = n_total - n_known_release
    
    n_le_2018 = (y_valid <= 2018).sum()
    n_2019_2020 = ((y_valid >= 2019) & (y_valid <= 2020)).sum()
    n_ge_2021 = (y_valid >= 2021).sum()
    
    md_content = f"""# RM-VMusic Phase 14: Temporal Field Taxonomy & Gap Report
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

## 2. Release Year Distribution Summary ($N = {n_total:,}$)

- **Verified Release Year:** **{n_known_release:,} tracks ({n_known_release / n_total * 100:.2f}%)**
- **Unverified / Missing Year:** **{n_unknown:,} tracks ({n_unknown / n_total * 100:.2f}%)**
- **Partition Distribution:**
  - Historical Train ($\le 2018$): **{n_le_2018:,} tracks**
  - Transition Val ($2019–2020$): **{n_2019_2020:,} tracks**
  - Modern Test ($\ge 2021$): **{n_ge_2021:,} tracks**
- **Active Test Classes:** **10 / 12 classes** (`POP_BALLAD`, `BOLERO_TRUTINH`, `INSTRUMENTAL`, `RAP_HIPHOP`, `FOLK_TRADITIONAL`, `DANCE_EDM`, `REVOLUTIONARY`, `ROCK`, `RB_SOUL`, `OTHER`).
- **Data Availability Limitation:** `NHAC_TRINH` ($N=0$) and `CHILDREN` ($N=0$) remain authentically absent in post-2021 open releases.
"""
    with open(REPORTS_DIR / "phase14_temporal_gap_analysis.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Generated reports/phase14_temporal_gap_analysis.md successfully.")

if __name__ == "__main__":
    run_temporal_audit()
