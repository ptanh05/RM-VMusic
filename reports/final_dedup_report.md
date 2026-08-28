# RM-VMusic: Final Exhaustive Deduplication Audit Report
**Audit Date:** 2026-08-28 13:14:59  
**Target Catalog:** `data/processed/final_12class_metadata.csv` ($N=5,515$)

---

## 1. Deduplication Verification Summary

| Identifier / Key Pair | Exact Duplicates | Allowed Tolerance | Verification Status |
|---|---|---|---|
| `song_id` | **0** | 0 | **CLEAN (PASSED)** |
| `(title, artist)` | **0** | 0 | **CLEAN (PASSED)** |
| Unicode Normalized Text NFC | **0 errors** | 0 | **CLEAN (PASSED)** |

---

## 2. Deduplication Policy & Implementation

1. **Deterministic Song ID:** Generated via SHA-256 hash over `source_id` and standardized `(title, artist)`.
2. **Artist Key Normalization:** Diacritics and casing are normalized using Unicode NFC before deduplication.
3. **Zero Cross-Source Duplication:** Tracks indexed across both `sunbv56` and `vietlyrics` are merged into single unified canonical entities with combined provenance tags.
