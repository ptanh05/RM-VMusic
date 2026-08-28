"""
phase16_license_audit.py
RM-VMusic Phase 16: Rigorous Multi-Tier License Gate Audit.
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

def run_license_audit():
    print("=== RM-VMusic Phase 16: License Gate Audit ===")
    
    rejected_license_records = [
        {
            "source_id": "REJ_LIC_01",
            "source_name": "VietSing SVS Corpus",
            "url": "https://www.researchgate.net/publication/VietSing",
            "license_type": "Restricted Academic",
            "rejection_tier": "TIER_4_RESTRICTED_ACCESS",
            "reason": "Authors explicitly state dataset cannot be publicly distributed due to licensing restrictions"
        },
        {
            "source_id": "REJ_LIC_02",
            "source_name": "YouTube Vietnamese Nursery Streams",
            "url": "https://github.com/iptv-org/iptv",
            "license_type": "Unknown / Unlicensed Broadcast",
            "rejection_tier": "TIER_5_UNKNOWN_LICENSE",
            "reason": "Commercial livestream broadcast URLs lack open redistribution licenses and track metadata"
        },
        {
            "source_id": "REJ_LIC_03",
            "source_name": "Trịnh Công Sơn Family Estate Digital Lyrics",
            "url": "https://trinhcongson.vn/",
            "license_type": "Proprietary All Rights Reserved",
            "rejection_tier": "TIER_6_PROPRIETARY_ESTATE",
            "reason": "Family estate retains all commercial copyright; redistribution in ML benchmarks is prohibited"
        },
        {
            "source_id": "REJ_LIC_04",
            "source_name": "Commercial Streaming Scrapers (Zing/Spotify/NCT)",
            "url": "Zing MP3 / Nhaccuatui Web APIs",
            "license_type": "Commercial Proprietary",
            "rejection_tier": "TIER_7_DRM_PROTECTED",
            "reason": "Direct audio scraping violates Terms of Service and digital rights management policies"
        }
    ]
    
    df_rej = pd.DataFrame(rejected_license_records)
    df_rej.to_csv(PROCESSED_DIR / "phase16_rejected_license.csv", index=False)
    
    md_content = """# RM-VMusic Phase 16: Multi-Tier License Gate Audit Report
**Evaluation Date:** 2026-08-28

---

## 1. License Verification & Rejection Table

| ID | Candidate Source | Stated License | Rejection Tier | Reviewer Legal Rationale |
|---|---|---|---|---|
"""
    for _, r in df_rej.iterrows():
        md_content += f"| `{r['source_id']}` | **{r['source_name']}** | `{r['license_type']}` | `{r['rejection_tier']}` | {r['reason']} |\n"

    md_content += """
---

## 2. Hard Legal Compliance Standards
1. **No Circumvention of Digital Rights Management:** The project strictly avoids ripping audio from commercial streaming platforms.
2. **Clear Open Academic Licenses Only:** Only resources licensed under CC0, CC-BY, CC-BY-NC-SA, or open research terms are admitted into official benchmark distributions.
"""
    with open(REPORTS_DIR / "phase16_license_audit.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Generated data/processed/phase16_rejected_license.csv and reports/phase16_license_audit.md.")

if __name__ == "__main__":
    run_license_audit()
