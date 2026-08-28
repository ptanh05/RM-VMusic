"""
phase13_dataset_audit.py
RM-VMusic Phase 13: Dataset Quality Scoring & Class-Wise Robustness Auditor.
"""
import sys
import os
import pandas as pd
import numpy as np
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

GENRES_12 = [
    "POP_BALLAD", "BOLERO_TRUTINH", "INSTRUMENTAL", "RAP_HIPHOP",
    "FOLK_TRADITIONAL", "DANCE_EDM", "REVOLUTIONARY", "NHAC_TRINH",
    "ROCK", "RB_SOUL", "OTHER", "CHILDREN"
]

def run_dataset_quality_audit():
    print("=== RM-VMusic Phase 13: Dataset Quality Scoring per Class ===")
    
    df_v2 = pd.read_csv(PROCESSED_DIR / "final_12class_metadata_v2.csv")
    
    quality_rows = []
    for g in GENRES_12:
        sub = df_v2[df_v2["genre"] == g]
        tot = len(sub)
        n_lyrics = (sub["lyrics_available"] == 1).sum() if "lyrics_available" in sub.columns else (sub["lyrics_status"] == "verified_local").sum()
        n_cover = (sub["cover_available"] == 1).sum() if "cover_available" in sub.columns else (sub["cover_status"] == "verified_local").sum()
        n_audio = 0
        y_valid = pd.to_numeric(sub["release_year"], errors="coerce").dropna().astype(int)
        n_year = len(y_valid)
        n_ge_2021 = (y_valid >= 2021).sum()
        n_artists = sub["artist"].nunique()
        
        # Calculate composite quality score / 100
        # Weights: Sample Support (25), Lyrics Coverage (25), Cover Coverage (15), Year Coverage (20), Modern Coverage (15)
        s_supp = min(25.0, (tot / 200.0) * 25.0)
        s_lyr = (n_lyrics / tot) * 25.0 if tot > 0 else 0
        s_cov = (n_cover / tot) * 15.0 if tot > 0 else 0
        s_yr = (n_year / tot) * 20.0 if tot > 0 else 0
        s_mod = min(15.0, (n_ge_2021 / 10.0) * 15.0)
        total_q = s_supp + s_lyr + s_cov + s_yr + s_mod
        
        if total_q >= 70:
            q_grade = "TIER_A (High Reliability)"
        elif total_q >= 45:
            q_grade = "TIER_B (Moderate Reliability)"
        else:
            q_grade = "TIER_C (Sparse Modality / Historic Only)"
            
        quality_rows.append({
            "Class": g,
            "Total_N": tot,
            "Lyrics_N": n_lyrics,
            "Lyrics_Pct": round((n_lyrics / tot) * 100.0, 2) if tot > 0 else 0,
            "Cover_N": n_cover,
            "Cover_Pct": round((n_cover / tot) * 100.0, 2) if tot > 0 else 0,
            "Audio_N": n_audio,
            "Audio_Pct": 0.0,
            "Year_Known_N": n_year,
            "Year_Known_Pct": round((n_year / tot) * 100.0, 2) if tot > 0 else 0,
            "N_ge_2021": n_ge_2021,
            "Artist_Count": n_artists,
            "Quality_Score": round(total_q, 1),
            "Quality_Grade": q_grade
        })
        print(f"  {g:<18}: N={tot:>4} | Lyrics={n_lyrics:>4} | Cover={n_cover:>3} | Year={n_year:>3} | >=2021={n_ge_2021:>2} | Score={total_q:>4.1f} ({q_grade})")
        
    df_q = pd.DataFrame(quality_rows)
    df_q.to_csv(REPORTS_DIR / "phase13_dataset_quality.csv", index=False)
    
    md_content = """# RM-VMusic Phase 13: Class-Wise Dataset Quality & Reliability Scorecard
**Evaluation Date:** 2026-08-28

---

## 1. Class Quality Scoring Table

| Genre Class | Samples ($N$) | Lyrics (%) | Cover (%) | Year Known (%) | Modern $\ge 2021$ | Quality Score (/100) | Reviewer Reliability Grade |
|---|---|---|---|---|---|---|---|
"""
    for _, r in df_q.iterrows():
        md_content += f"| `{r['Class']}` | {r['Total_N']:,} | {r['Lyrics_Pct']}% | {r['Cover_Pct']}% | {r['Year_Known_Pct']}% | **{r['N_ge_2021']:,}** | **{r['Quality_Score']}** | {r['Quality_Grade']} |\n"

    md_content += """
---

## 2. Quality Tier Definitions
- **Tier A (Score $\ge 70$):** High sample support, rich linguistic lyrics coverage, verified chronological representation.
- **Tier B (Score $45–69$):** Solid core annotations; moderate modality sparsity.
- **Tier C (Score $< 45$):** Historical or extreme niche genres (e.g., `NHAC_TRINH`, `CHILDREN`, `OTHER`) where release years are vintage or lyrics/covers are sparse.
"""
    with open(REPORTS_DIR / "phase13_dataset_quality.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Generated reports/phase13_dataset_quality.csv and reports/phase13_dataset_quality.md.")

if __name__ == "__main__":
    run_dataset_quality_audit()
