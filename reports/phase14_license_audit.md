# RM-VMusic Phase 14: License Gate & Copyright Audit
**Evaluation Date:** 2026-08-28

---

## 1. Multi-Tier Legal Classification Table

| Source Identifier | License Standard | Academic Research Rights | Commercial Rights | Redistribution Risk | Final Gate Decision |
|---|---|---|---|---|---|
| `VietLyrics (`tsdocode/vi-song-7k-public`)` | CC-BY-NC-SA 4.0 | PERMITTED (Attribution required) | PROHIBITED | LOW | **`APPROVED_ACADEMIC_USAGE`** |
| `VNTM Traditional Music (Kaggle LTPhat)` | CC0 / Public Domain | PERMITTED (Unrestricted) | PERMITTED | NONE | **`APPROVED_OPEN_DATA`** |
| `sunbv56 / Song Dataset` | Open Academic Research | PERMITTED | RESTRICTED | LOW | **`APPROVED_RESEARCH_ONLY`** |
| `Unlicensed GitHub Scraper Repositories` | Unknown / Unspecified | UNCERTAIN / HIGH LEGAL RISK | PROHIBITED | HIGH | **`REJECTED_LICENSE_UNKNOWN`** |
| `Commercial Streaming Services (Zing/NCT/Spotify)` | Proprietary Commercial | RESTRICTED | STRICTLY PROHIBITED | CRITICAL (Copyright & DRM boundary) | **`REJECTED_COMMERCIAL`** |

---

## 2. Hard Legal Constraints Enforced
1. **Public Availability != Legal Reusability:** Just because a student script or repository is visible publicly on GitHub does not grant legal distribution rights for research benchmarks without explicit open licenses (e.g., CC0, CC-BY, MIT).
2. **Zero Commercial Scraping:** No audio waveforms or copyrighted commercial lyrics were scraped from streaming platforms to falsely inflate dataset numbers.
