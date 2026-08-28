# RM-VMusic Phase 13: External Sources & Acquisition Audit
**Evaluation Date:** 2026-08-28

---

## 1. Multi-Repository External Discovery Table

| Dataset Identifier | URL / Provider | License | Records | Vietnamese Coverage | Audio / Lyrics | Research Usability | Final Reviewer Status |
|---|---|---|---|---|---|---|---|
| `VietLyrics (tsdocode/vi-song-7k-public)` | [https://huggingface.co/datasets/tsdocode/vi-song-7k-public](https://huggingface.co/datasets/tsdocode/vi-song-7k-public) | CC-BY-NC-SA 4.0 | 8,428 | 100% | Lyrics: Yes (7,433) / Audio: No (Commercial streaming links only) | Yes (Non-commercial academic) | **ACCEPTED (Primary Ground Truth Catalog)** |
| `sunbv56/song_dataset` | [https://huggingface.co/datasets/sunbv56/song_dataset](https://huggingface.co/datasets/sunbv56/song_dataset) | Open Academic Research | 9,344 | 100% | Lyrics: Yes (9,344) / Audio: No (CDN streaming links expired) | Yes (Research only) | **ACCEPTED (Lyrics & Metadata Pool)** |
| `Vietnam Traditional Music (VNTM / LTPhat)` | [https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres](https://www.kaggle.com/datasets/homata123/vntm-for-building-model-5-genres) | CC0 / Public Domain | 1,250 | 100% | Lyrics: No / Audio: Yes (Short audio clips) | Yes (Public Domain) | **ACCEPTED FOR FOLK_TRADITIONAL EXTENSION** |
| `Vietnamese Music Dataset (Toan-Minh-Duong-Son)` | [https://huggingface.co/datasets/Toan-Minh-Duong-Son/vietnamese-music-dataset](https://huggingface.co/datasets/Toan-Minh-Duong-Son/vietnamese-music-dataset) | Unknown / Unspecified | 450 | 100% | Lyrics: No / Audio: Yes | No (Unclear legal rights) | **REJECTED (License Ambiguity)** |
| `Zing MP3 / Nhaccuatui Streaming Indexes` | [https://zingmp3.vn](https://zingmp3.vn) | Commercial Proprietary | 100,000 | 100% | Lyrics: Proprietary / Audio: Commercial DRM stream | No | **REJECTED (Commercial DRM & Copyright Protection)** |

---

## 2. Key Findings on Targeted Genres
1. **Vietnamese Children's Songs (`CHILDREN`):** No independent open-access dataset dedicated to Vietnamese children songs exists on Hugging Face, Kaggle, GitHub, or Zenodo. All available tracks stem from general crawls where post-2021 release dates are absent.
2. **Nhạc Trịnh (`NHAC_TRINH`):** No open dataset indexes post-2021 recordings of Trịnh Công Sơn compositions with verified release dates.
3. **Traditional Music (`FOLK_TRADITIONAL`):** The VNTM dataset (Kaggle/GitHub LTPhat) provides high-quality open traditional audio recordings under CC0 license.
