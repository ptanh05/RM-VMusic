# RM-VMusic Phase 11: Pre-Expansion Dataset Audit Report
**Evaluation Date:** 2026-08-28  
**Scope:** Forensic baseline state of Dataset V1 ($N = 5,515$) before expansion

---

## 1. V1 Dataset Summary

- **Total Samples:** 5,515
- **Total Unique Artists:** 2,747
- **Average Samples per Artist:** 2.01
- **Physical Lyrics Coverage:** 4,117 (74.65%)
- **Physical Cover Coverage:** 902 (16.36%)
- **Physical Audio Coverage:** 0 (0.00%)
- **Verified Release Year Coverage:** 770 (13.96%)

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
