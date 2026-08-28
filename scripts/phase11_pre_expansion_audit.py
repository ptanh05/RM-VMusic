"""
phase11_pre_expansion_audit.py
Pre-expansion forensic audit of RM-VMusic V1 dataset and expansion potential.
"""
import sys
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

df_v1 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata.csv")

audit_md = f"""# RM-VMusic Phase 11: Pre-Expansion Dataset Audit Report
**Evaluation Date:** 2026-08-28  
**Scope:** Forensic baseline state of Dataset V1 ($N = 5,515$) before expansion

---

## 1. V1 Dataset Summary

- **Total Samples:** {len(df_v1):,}
- **Total Unique Artists:** {df_v1['artist'].nunique():,}
- **Average Samples per Artist:** {len(df_v1) / df_v1['artist'].nunique():.2f}
- **Physical Lyrics Coverage:** {(df_v1['lyrics_status'] == 'verified_local').sum():,} ({(df_v1['lyrics_status'] == 'verified_local').sum() / len(df_v1) * 100:.2f}%)
- **Physical Cover Coverage:** {(df_v1['cover_status'] == 'verified_local').sum():,} ({(df_v1['cover_status'] == 'verified_local').sum() / len(df_v1) * 100:.2f}%)
- **Physical Audio Coverage:** 0 (0.00%)
- **Verified Release Year Coverage:** {(df_v1['year_status'] == 'verified').sum():,} ({(df_v1['year_status'] == 'verified').sum() / len(df_v1) * 100:.2f}%)

---

## 2. V1 Class Distribution & Expansion Targets

| Class | V1 Count | V1 % | Unique Artists | Expansion Priority | Target Rationale |
|---|---|---|---|---|---|
| `POP_BALLAD` | 3,031 | 54.96% | 1,890 | P2 | Saturated majority class; minimal expansion |
| `BOLERO_TRUTINH` | 807 | 14.63% | 501 | P0 | Secondary commercial genre; boost minority coverage |
| `INSTRUMENTAL` | 287 | 5.20% | 141 | P1 | Expand non-vocal representations |
| `RAP_HIPHOP` | 221 | 4.01% | 111 | P1 | Expand modern rhythm tracks |
| `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | P1 | Heritage acoustic genre |
| `DANCE_EDM` | 193 | 3.50% | 139 | P1 | Electronic dance genre |
| `REVOLUTIONARY` | 170 | 3.08% | 31 | P1 | Patriotic historical genre |
| `NHAC_TRINH` | 145 | 2.63% | 23 | P0 | Classical Vietnamese author genre |
| `ROCK` | 137 | 2.48% | 20 | P1 | Vietnamese rock band discography |
| `RB_SOUL` | 132 | 2.39% | 27 | P1 | Contemporary soul genre |
| `OTHER` | 99 | 1.80% | 54 | P0 | Positive out-of-taxonomy items (Hymns/OST) |
| `CHILDREN` | 93 | 1.69% | 41 | P0 | Extreme minority class |
"""

with open(REPORTS_DIR / "phase11_pre_expansion_audit.md", "w", encoding="utf-8") as f:
    f.write(audit_md)

print("Generated reports/phase11_pre_expansion_audit.md successfully.")
