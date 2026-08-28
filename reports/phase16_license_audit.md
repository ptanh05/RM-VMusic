# RM-VMusic Phase 16: Multi-Tier License Gate Audit Report
**Evaluation Date:** 2026-08-28

---

## 1. License Verification & Rejection Table

| ID | Candidate Source | Stated License | Rejection Tier | Reviewer Legal Rationale |
|---|---|---|---|---|
| `REJ_LIC_01` | **VietSing SVS Corpus** | `Restricted Academic` | `TIER_4_RESTRICTED_ACCESS` | Authors explicitly state dataset cannot be publicly distributed due to licensing restrictions |
| `REJ_LIC_02` | **YouTube Vietnamese Nursery Streams** | `Unknown / Unlicensed Broadcast` | `TIER_5_UNKNOWN_LICENSE` | Commercial livestream broadcast URLs lack open redistribution licenses and track metadata |
| `REJ_LIC_03` | **Trịnh Công Sơn Family Estate Digital Lyrics** | `Proprietary All Rights Reserved` | `TIER_6_PROPRIETARY_ESTATE` | Family estate retains all commercial copyright; redistribution in ML benchmarks is prohibited |
| `REJ_LIC_04` | **Commercial Streaming Scrapers (Zing/Spotify/NCT)** | `Commercial Proprietary` | `TIER_7_DRM_PROTECTED` | Direct audio scraping violates Terms of Service and digital rights management policies |

---

## 2. Hard Legal Compliance Standards
1. **No Circumvention of Digital Rights Management:** The project strictly avoids ripping audio from commercial streaming platforms.
2. **Clear Open Academic Licenses Only:** Only resources licensed under CC0, CC-BY, CC-BY-NC-SA, or open research terms are admitted into official benchmark distributions.
