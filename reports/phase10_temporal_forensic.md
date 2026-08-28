# RM-VMusic Phase 10: Deep Temporal Metadata & Class Absence Forensic Audit Report
**Audit Date:** 2026-08-28  
**Audit Scope:** Full Investigation into Release Year Metadata ($N=5,515$) and Temporal Partition Integrity

---

## 1. Executive Summary & Core Discovery

The forensic audit of release year metadata reveals that the absence of `NHAC_TRINH` ($N=0$), `CHILDREN` ($N=0$), and sparsity of `OTHER` ($N=2$) and `BOLERO_TRUTINH` ($N=7$) in the temporal test set ($\ge 2021$) is driven by a combination of **historical genre life cycles** and **severe upstream metadata selection bias**:

1. **Massive Upstream Selection Bias:** While only $13.96\%$ ($770 / 5,515$) of all tracks contain verified release years, year tags are heavily concentrated in specific curated genres (`ROCK`: $87.6\%$, `RB_SOUL`: $84.9\%$, `NHAC_TRINH`: $66.2\%$, `REVOLUTIONARY`: $64.1\%$). In contrast, `POP_BALLAD` ($1.15\%$) and `BOLERO_TRUTINH` ($0.99\%$) almost entirely lack release year metadata in raw crawls.
2. **Historical Life Cycle Effect:**
   - **`NHAC_TRINH`:** Trịnh Công Sơn's discography is fundamentally vintage (1960s–1990s). In our verified dataset, $95$ tracks are dated $\le 2018$ and $0$ tracks exist in $\ge 2021$.
   - **`CHILDREN`:** All $12$ verified children's songs belong to the 2004–2008 era ($\le 2018$), with $0$ verified post-2021 tracks.
   - **`OTHER`:** 97 of 99 tracks are sacred/religious hymns without commercial release year tags; only 2 tracks (2021 OST singles) contain release years, leaving **0 training tracks $\le 2018$**.

---

## 2. Release Year Coverage per Class ($N=5,515$)

| Class | Total Samples | Has Verified Year | Missing Year | Year Coverage (%) | Metadata Bias |
|---|---|---|---|---|---|
| `ROCK` | 137 | **120** | 17 | **87.59%** | Strongly Over-Represented |
| `RB_SOUL` | 132 | **112** | 20 | **84.85%** | Strongly Over-Represented |
| `NHAC_TRINH` | 145 | **96** | 49 | **66.21%** | Strongly Over-Represented |
| `REVOLUTIONARY` | 170 | **109** | 61 | **64.12%** | Strongly Over-Represented |
| `RAP_HIPHOP` | 221 | **106** | 115 | **47.96%** | Moderate Coverage |
| `FOLK_TRADITIONAL`| 200 | **88** | 112 | **44.00%** | Moderate Coverage |
| `DANCE_EDM` | 193 | **40** | 153 | **20.73%** | Under-Represented |
| `INSTRUMENTAL` | 287 | **42** | 245 | **14.63%** | Under-Represented |
| `CHILDREN` | 93 | **12** | 81 | **12.90%** | Heavily Missing |
| `OTHER` | 99 | **2** | 97 | **2.02%** | Extremely Missing |
| `POP_BALLAD` | 3,031 | **35** | 2,996 | **1.15%** | Massive Metadata Absence |
| `BOLERO_TRUTINH` | 807 | **8** | 799 | **0.99%** | Massive Metadata Absence |

---

## 3. Chronological Distribution Across Partitions

| Class | Min Year | Max Year | Median Year | Train ($\le 2018$) | Val ($2019-2020$) | Test ($\ge 2021$) | Missing |
|---|---|---|---|---|---|---|---|
| `POP_BALLAD` | 2006 | 2024 | 2020.0 | 11 | 8 | 16 | 2,996 |
| `BOLERO_TRUTINH` | 2000 | 2024 | 2023.0 | 1 | 0 | 7 | 799 |
| `INSTRUMENTAL` | 1991 | 2025 | 2003.0 | 36 | 2 | 4 | 245 |
| `RAP_HIPHOP` | 1983 | 2026 | 2022.0 | 37 | 11 | **58** | 115 |
| `FOLK_TRADITIONAL`| 1989 | 2024 | 2007.0 | 77 | 2 | 9 | 112 |
| `DANCE_EDM` | 1996 | 2025 | 2020.0 | 16 | 6 | 18 | 153 |
| `REVOLUTIONARY` | 1997 | 2024 | 2010.0 | 101 | 3 | 5 | 61 |
| `NHAC_TRINH` | 1974 | 2020 | 2003.0 | **95** | 1 | **0 (Missing)** | 49 |
| `ROCK` | 1997 | 2024 | 2009.5 | 92 | 15 | 13 | 17 |
| `RB_SOUL` | 1967 | 2026 | 2021.0 | 48 | 6 | **58** | 20 |
| `OTHER` | 2022 | 2022 | 2022.0 | **0 (Missing)**| 0 | 2 | 97 |
| `CHILDREN` | 2004 | 2008 | 2004.0 | 12 | 0 | **0 (Missing)** | 81 |
| **TOTAL** | **1967** | **2026** | **2014.0** | **526** | **54** | **190** | **4,745** |

---

## 4. Class Prior Comparison & Selection Bias

| Class | Full Dataset Prior (%) | Known-Year Prior (%) | Temporal Train Prior (%) | Temporal Test Prior (%) | Selection Bias (Known - Full) |
|---|---|---|---|---|---|
| `POP_BALLAD` | 54.96% | 4.55% | 2.09% | 8.42% | **-50.41%** (Under-sampled) |
| `BOLERO_TRUTINH` | 14.63% | 1.04% | 0.19% | 3.68% | **-13.59%** (Under-sampled) |
| `ROCK` | 2.48% | 15.58% | 17.49% | 6.84% | **+13.10%** (Over-sampled) |
| `RB_SOUL` | 2.39% | 14.55% | 9.13% | 30.53% | **+12.16%** (Over-sampled) |
| `REVOLUTIONARY` | 3.08% | 14.16% | 19.20% | 2.63% | **+11.08%** (Over-sampled) |
| `NHAC_TRINH` | 2.63% | 12.47% | 18.06% | 0.00% | **+9.84%** (Over-sampled) |
| `RAP_HIPHOP` | 4.01% | 13.77% | 7.03% | 30.53% | **+9.76%** (Over-sampled) |
| `FOLK_TRADITIONAL`| 3.63% | 11.43% | 14.64% | 4.74% | **+7.80%** (Over-sampled) |

---

## 5. Alternative Temporal Protocols Comparison

1. **Protocol A (Current Standard):** Train $\le 2018$ ($N=526$), Val $2019-2020$ ($N=54$), Test $\ge 2021$ ($N=190$).
   - *Active classes in Test:* **10 / 12 classes** (`NHAC_TRINH` and `CHILDREN` absent from Test).
   - *Status:* **FEASIBLE & STANDARDIZED**.
2. **Protocol B:** Train $\le 2020$ ($N=580$), Test $\ge 2021$ ($N=190$).
   - *Impact:* Merges Validation into Train; retains identical test set with 10 active classes.
3. **Protocol C (Active-Classes-Only Closed Evaluation):** Evaluates strictly the 9 common classes present in both Train and Test ($N_{\text{train}}=419, N_{\text{test}}=188$).
   - *Impact:* Eliminates zero-recall penalty on missing classes, providing an un-penalized view of temporal feature drift.
