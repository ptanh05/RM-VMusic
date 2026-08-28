# RM-VMusic Phase 13: Legal & Intellectual Property Audit
**Evaluation Date:** 2026-08-28

---

## 1. License Classification & Redistribution Protocol

| Source Repository | Stated License | Academic Redistribution | Commercial Rights | Compliance Determination |
|---|---|---|---|---|
| `VietLyrics (`vi-song-7k-public`)` | `CC-BY-NC-SA 4.0` | Permitted for Non-Commercial Academic Research | Prohibited without custom authorization | **100% COMPLIANT (Academic Benchmark Usage)** |
| `sunbv56 / Song Dataset` | `Open Academic Research` | Permitted for Academic Research & NLP benchmarks | Restricted | **100% COMPLIANT (Academic NLP/MIR Alignment)** |
| `Vietnam Traditional Music (VNTM)` | `CC0 / Public Domain` | Permitted without restrictions | Permitted | **100% COMPLIANT (Open Data)** |
| `Zing MP3 Stream Platform` | `Proprietary Commercial` | Strictly Prohibited | Proprietary | **EXCLUDED FROM DIRECT AUDIO CRAWL (Zero-Masking Enforced)** |

---

## 2. Ethical Data Principles
1. **No Circumvention of Access Controls:** The project strictly avoids bypassing HMAC authentication tokens, DRM schemes, or proprietary streaming encryption.
2. **Attribution & Provenance:** Every sample incorporated into the dataset retains full source identifier metadata (`source`, `source_id`, `label_source`).
3. **Zero-Masking as Legal Safety Standard:** Where raw audio waveforms cannot be legally distributed under open academic licenses, audio is represented as a clean zero-vector ($mask=0.0$), enabling the evaluation of missing modality robustness without copyright infringement.
