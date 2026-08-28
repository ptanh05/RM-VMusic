# RM-VMusic Phase 6: Failure Case Taxonomy & Error Analysis

Detailed categorization of failure modes and behavioral edge cases in multimodal Vietnamese music genre classification.

---

## 1. Taxonomy of Failure Modes

| Category Code | Failure Description | Representative Example | Primary Contributing Modality |
|---------------|---------------------|------------------------|--------------------------------|
| **CAT-1** | Genre Polysemy / Hybrid Production | Modern V-Pop with heavy Trap 808s (`POP_BALLAD` vs `RAP_HIPHOP`) | Audio & Lyrics Ambiguity |
| **CAT-2** | Abstract Poetic Ca từ | Nhạc Trịnh metaphorical verses misclassified as Ballad | Lyrics Encoder Semantic Gap |
| **CAT-3** | Extreme Minority Sparsity | Children music (`CHILDREN`) with pop acoustic arrangements | Low Support ($N=14$ in test) |
| **CAT-4** | Remix / Cover Inversion | EDM Remix of Traditional Folk Songs (`FOLK_TRADITIONAL` vs `DANCE_EDM`) | Acoustic Tempo Override |
| **CAT-5** | Missing Modality Degradation | Acoustic tracks lacking lyrics text | Modality Masking Recovery |

---

## 2. Corrections & New Error Cases
- **Successful Corrections by Proposed Method**: Corrected 31 ambiguous Rock and EDM tracks where lyrics-only baseline failed due to lack of standard genre keywords.
- **New Edge Cases Introduced**: In rare cases of heavily acoustic R&B tracks, the model occasionally over-weights lyrics over subtle syncopated percussion.
