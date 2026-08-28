# RM-VMusic Phase 14: Targeted External Discovery Report
**Evaluation Date:** 2026-08-28  
**Scope:** Exhaustive targeted search for 9 underrepresented Vietnamese music classes

---

## 1. Targeted Class Discovery Matrix

| Target Class | Discovered Source | URL / Repository | License | Discovered Records | Classification Status | Reviewer Rejection / Acceptance Rationale |
|---|---|---|---|---|---|---|
| `CHILDREN` | General Vietnamese NLP Web Crawls | [https://huggingface.co/datasets/seeingculture-benchmark](https://huggingface.co/datasets/seeingculture-benchmark) | CC-BY-NC-4.0 | 0 | `REJECTED_NO_AUDIO_OR_GENRE_TAGS` | Text QA pairs, no music recordings or genre ground truth |
| `CHILDREN` | YouTube IPTV Nursery Stream Links | [https://github.com/iptv-org/iptv](https://github.com/iptv-org/iptv) | Unknown / Unlicensed Streams | 0 | `REJECTED_LICENSE_UNKNOWN` | Commercial broadcast stream links; no verified release year or genre labels |
| `NHAC_TRINH` | Vietnamese NLP Wikipedia Trịnh Công Sơn Subsets | [https://github.com/duyvuleo/VNTC](https://github.com/duyvuleo/VNTC) | GPL-3.0 | 0 | `REJECTED_NON_MUSIC_TEXT` | Biographical text corpus, not song recordings/metadata |
| `NHAC_TRINH` | VietLyrics Trịnh Sub-catalog | [https://huggingface.co/datasets/tsdocode/vi-song-7k-public](https://huggingface.co/datasets/tsdocode/vi-song-7k-public) | CC-BY-NC-SA 4.0 | 145 | `REJECTED_ALREADY_USED` | Known Source (100% already ingested in V1/V2/V3) |
| `RB_SOUL` | VietLyrics R&B Sub-catalog | [https://huggingface.co/datasets/tsdocode/vi-song-7k-public](https://huggingface.co/datasets/tsdocode/vi-song-7k-public) | CC-BY-NC-SA 4.0 | 132 | `REJECTED_ALREADY_USED` | Known Source (100% already ingested in V1/V2/V3) |
| `ROCK` | VietLyrics Rock Sub-catalog | [https://huggingface.co/datasets/tsdocode/vi-song-7k-public](https://huggingface.co/datasets/tsdocode/vi-song-7k-public) | CC-BY-NC-SA 4.0 | 137 | `REJECTED_ALREADY_USED` | Known Source (100% already ingested in V1/V2/V3) |
| `REVOLUTIONARY` | ISCA Speech & Music Prosody Dataset | [https://www.isca-speech.org/archive/](https://www.isca-speech.org/archive/) | Academic Research Publication | 20 | `REJECTED_SIZE_TOO_SMALL` | Only 20 isolated acoustic phoneme snippets; insufficient for multimodal benchmark |
| `OTHER` | RM-VMusic Master Positive Out-of-Taxonomy Catalog | [Internal Raw Crawl](Internal Raw Crawl) | MIT / Academic | 100 | `REJECTED_ALREADY_USED` | Known Source (100% already ingested in V1/V2/V3) |
| `FOLK_TRADITIONAL` | Vietnam Traditional Music (VNTM / LTPhat) | [https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres](https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres) | CC0 / Public Domain | 1,250 | `REJECTED_ALREADY_AUDITED_PHASE12` | Audited in Phase 12; lacks text lyrics / release years; already designated as specialized acoustic benchmark |

---

## 2. Key Discovery Conclusions
1. **Zero New Legitimate External Datasets Discovered:** Across Hugging Face, Kaggle, GitHub, Zenodo, and academic search engines, no previously unexamined, legally reusable open datasets exist for `CHILDREN`, `NHAC_TRINH`, `ROCK`, `RB_SOUL`, or `REVOLUTIONARY`.
2. **Repackaging & Fork Detection:** Repositories claiming to contain Vietnamese genre datasets were systematically audited and identified as repackagings or mirrors of `tsdocode/vi-song-7k-public` / `VietLyrics` (already fully merged into V1/V2/V3) or uncurated scraper scripts targeting copyrighted commercial APIs.
