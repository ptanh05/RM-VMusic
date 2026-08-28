# RM-VMusic: Forensic Dataset & Split Action Plan
**Evaluation Date:** 2026-08-28

---

## 1. Scientific Severity Summary

| Severity | Issue Description | Impact on Research | Recommended Fix |
|---|---|---|---|
| 🔴 **CRITICAL** | **Temporal Split Missing Classes:** `NHAC_TRINH` ($N=0$) and `CHILDREN` ($N=0$) are missing from Temporal Test; `OTHER` ($N=0$) is missing from Temporal Train. | Severe artificial penalty on 12-class Macro-F1 evaluation under chronological partitioning. | **P0:** Explicitly document the 10-class active test space in temporal evaluation and compute both 12-class standard Macro-F1 and 10-class Active Macro-F1. |
| 🟠 **MAJOR** | **Zero Lyrics Coverage for Class `OTHER`:** None of the 99 OTHER tracks contain physical lyrics. | Model must rely entirely on cover moments and prior probabilities for `OTHER`. | **P1:** Document `OTHER` as an extreme missing-modality test case for linguistic fallback. |
| 🟠 **MAJOR** | **Natural Market Imbalance ($Gini = 0.6102$):** 32.59x ratio between `POP_BALLAD` and minority classes. | Minority class recall remains low across all architectures. | **P1:** Maintain Balanced Cross-Entropy and report Balanced Accuracy alongside Macro-F1. |
| 🟡 **MINOR** | **Generic Streaming Placeholder Covers:** 2 default website images appear across distinct tracks. | Negligible impact on classification. | **P2:** Filter default placeholder image hashes in future dataset revisions. |
| 🟢 **PASS** | **IID & Artist Disjoint Splits:** Mathematically proven 0% artist leakage, 0 duplicates, Train-only TF-IDF. | High experimental integrity. | Maintain current pipeline. |

---

## 2. Prioritized Action Plan

### 🔴 P0 (Mandatory & Immediate)
1. **Temporal Evaluation Transparency:** Clearly document in all temporal tables that the temporal test set contains 10 active classes (190 tracks).
2. **Explicit Zero-Masking Definition:** Clearly define that audio is represented by zero-vectors with binary mask $mask = 0.0$.

### 🟠 P1 (Important for Extended Study)
3. **Active Class Macro-F1 Metric:** When reporting temporal shift, report both the standard 12-class Macro-F1 and the 10-class active-set Macro-F1 to give a complete picture of temporal generalization.
4. **Collect Additional Contemporary Lyrics:** Search open community repositories for lyrics of positive `OTHER` (hymns/soundtracks) tracks.

### 🟡 P2 (Optional Quality Improvements)
5. **Placeholder Image Pruning:** Identify and prune the 2 upstream default placeholder image hashes from the cover feature cache.
