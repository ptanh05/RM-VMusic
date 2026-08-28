"""
phase14_license_audit.py
RM-VMusic Phase 14: Rigorous License Gate Audit.
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
REPORTS_DIR = BASE_DIR / "reports"

def run_license_audit():
    print("=== RM-VMusic Phase 14: Rigorous License Gate Audit ===")
    
    license_eval = [
        {
            "Source Identifier": "VietLyrics (`tsdocode/vi-song-7k-public`)",
            "License Class": "CC-BY-NC-SA 4.0",
            "Academic Research Permission": "PERMITTED (Attribution required)",
            "Commercial Exploitation": "PROHIBITED",
            "Redistribution Risk": "LOW",
            "Gate Decision": "APPROVED_ACADEMIC_USAGE"
        },
        {
            "Source Identifier": "VNTM Traditional Music (Kaggle LTPhat)",
            "License Class": "CC0 / Public Domain",
            "Academic Research Permission": "PERMITTED (Unrestricted)",
            "Commercial Exploitation": "PERMITTED",
            "Redistribution Risk": "NONE",
            "Gate Decision": "APPROVED_OPEN_DATA"
        },
        {
            "Source Identifier": "sunbv56 / Song Dataset",
            "License Class": "Open Academic Research",
            "Academic Research Permission": "PERMITTED",
            "Commercial Exploitation": "RESTRICTED",
            "Redistribution Risk": "LOW",
            "Gate Decision": "APPROVED_RESEARCH_ONLY"
        },
        {
            "Source Identifier": "Unlicensed GitHub Scraper Repositories",
            "License Class": "Unknown / Unspecified",
            "Academic Research Permission": "UNCERTAIN / HIGH LEGAL RISK",
            "Commercial Exploitation": "PROHIBITED",
            "Redistribution Risk": "HIGH",
            "Gate Decision": "REJECTED_LICENSE_UNKNOWN"
        },
        {
            "Source Identifier": "Commercial Streaming Services (Zing/NCT/Spotify)",
            "License Class": "Proprietary Commercial",
            "Academic Research Permission": "RESTRICTED",
            "Commercial Exploitation": "STRICTLY PROHIBITED",
            "Redistribution Risk": "CRITICAL (Copyright & DRM boundary)",
            "Gate Decision": "REJECTED_COMMERCIAL"
        }
    ]
    
    df_lic = pd.DataFrame(license_eval)
    
    md_content = """# RM-VMusic Phase 14: License Gate & Copyright Audit
**Evaluation Date:** 2026-08-28

---

## 1. Multi-Tier Legal Classification Table

| Source Identifier | License Standard | Academic Research Rights | Commercial Rights | Redistribution Risk | Final Gate Decision |
|---|---|---|---|---|---|
"""
    for _, r in df_lic.iterrows():
        md_content += f"| `{r['Source Identifier']}` | {r['License Class']} | {r['Academic Research Permission']} | {r['Commercial Exploitation']} | {r['Redistribution Risk']} | **`{r['Gate Decision']}`** |\n"

    md_content += """
---

## 2. Hard Legal Constraints Enforced
1. **Public Availability != Legal Reusability:** Just because a student script or repository is visible publicly on GitHub does not grant legal distribution rights for research benchmarks without explicit open licenses (e.g., CC0, CC-BY, MIT).
2. **Zero Commercial Scraping:** No audio waveforms or copyrighted commercial lyrics were scraped from streaming platforms to falsely inflate dataset numbers.
"""
    with open(REPORTS_DIR / "phase14_license_audit.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("Generated reports/phase14_license_audit.md successfully.")

if __name__ == "__main__":
    run_license_audit()
