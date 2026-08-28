# RM-VMusic Phase 7B: Physical Lyrics Audit & Manifest Report
**Audit Date:** 2026-08-28 13:07:11  
**Total Records Processed:** 8,738 (Trainable: 5,416)  
**Status:** Complete Linguistic & Integrity Audit

---

## 1. Executive Summary

- **Total Valid Physical Lyrics on Disk:** **4,117 / 8,738** (47.12%)
- **Trainable Set Physical Lyrics:** **4,117 / 5,416** (**76.02% coverage**)
- **Missing Lyrics (Trainable Set):** 1,299 (23.98%)
- **Empty / 0-byte Files:** 0
- **Average Word Count (Valid Tracks):** 368.0 words
- **Average Character Count:** 1575.5 characters

---

## 2. Genre-Level Lyrics Coverage (Trainable Set)

| Genre Class | Total Tracks | With Physical Lyrics | Coverage % | Mean Words |
|---|---|---|---|---|
| `BOLERO_TRUTINH` | 807 | 694 | 86.00% | 325.3 |
| `CHILDREN` | 93 | 74 | 79.57% | 284.1 |
| `DANCE_EDM` | 193 | 149 | 77.20% | 388.8 |
| `FOLK_TRADITIONAL` | 200 | 82 | 41.00% | 317.9 |
| `INSTRUMENTAL` | 287 | 217 | 75.61% | 294.8 |
| `NHAC_TRINH` | 145 | 12 | 8.28% | 330.2 |
| `POP_BALLAD` | 3,031 | 2,726 | 89.94% | 379.8 |
| `RAP_HIPHOP` | 221 | 111 | 50.23% | 581.8 |
| `RB_SOUL` | 132 | 14 | 10.61% | 432.1 |
| `REVOLUTIONARY` | 170 | 23 | 13.53% | 290.1 |
| `ROCK` | 137 | 15 | 10.95% | 260.4 |

---

## 3. Linguistic Integrity & Formatting Standards

1. **Unicode NFC Normalization:** All text files in `data/lyrics/` adhere to standard NFC UTF-8 encoding.
2. **Missing Modality Representation:** Songs lacking lyrics are flagged with `local_path = ""` and receive zero-masks in downstream multimodal encoders.
