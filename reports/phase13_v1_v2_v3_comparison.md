# RM-VMusic Phase 13: Master Dataset Evolution (V1 $\to$ V2 $\to$ V3)
**Evaluation Date:** 2026-08-28

---

## 1. System Dimension Comparison

| Metric / Dimension | Dataset V1 (Phase 7-9 Baseline) | Dataset V2 (Phase 11 Expansion) | Dataset V3 (Phase 13 Legitimate Recovery) |
|---|---|---|---|
| **Total Track Samples** | 5,515 | 5,569 | **5,569** |
| **Unique Artists** | 2,747 | 2,770 | **2,770** |
| **Physical Lyrics Files** | 4,117 (74.65%) | 4,171 (74.89%) | **4,171 (74.89%)** |
| **Physical Cover Art** | 902 (16.36%) | 902 (16.20%) | **902 (16.20%)** |
| **Physical Audio Waveforms**| 0 (0.00% - Zero-Masked) | 0 (0.00% - Zero-Masked) | **0 (0.00% - Zero-Masked)** |
| **Verified Release Years** | 770 (13.96%) | 770 (13.83%) | **770 (13.83%)** |
| **Temporal Test Active Classes**| 10 / 12 classes | 10 / 12 classes | **10 / 12 classes** |
| **Duplicate IDs** | 0 | 0 | **0 (100% Unique)** |
| **Artist Leakage (AD Split)**| 0.00% (Strictly 0) | 0.00% (Strictly 0) | **0.00% (Strictly 0)** |

---

## 2. Per-Class Evolution Table

| Genre Class | V1 Count | V2 Count | V3 Count | Total Gain (Δ) |
|---|---|---|---|---|
| `POP_BALLAD` | 3,031 | 3,072 | 3,072 | **+41** |
| `BOLERO_TRUTINH` | 807 | 814 | 814 | **+7** |
| `INSTRUMENTAL` | 287 | 289 | 289 | **+2** |
| `RAP_HIPHOP` | 221 | 221 | 221 | **+0** |
| `FOLK_TRADITIONAL` | 200 | 200 | 200 | **+0** |
| `DANCE_EDM` | 193 | 196 | 196 | **+3** |
| `REVOLUTIONARY` | 170 | 170 | 170 | **+0** |
| `NHAC_TRINH` | 145 | 145 | 145 | **+0** |
| `ROCK` | 137 | 137 | 137 | **+0** |
| `RB_SOUL` | 132 | 132 | 132 | **+0** |
| `OTHER` | 99 | 100 | 100 | **+1** |
| `CHILDREN` | 93 | 93 | 93 | **+0** |
