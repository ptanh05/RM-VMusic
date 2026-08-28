# RM-VMusic Phase 15: Cross-Source Semantic Label Mapping Specification
**Evaluation Date:** 2026-08-28

---

## 1. Ground Truth Semantic Mapping Table

| Original Source Label | RM-VMusic 12 Target Class | Mapping Rationale | Confidence Level |
|---|---|---|---|
| `Nhạc thiếu nhi / Ca khúc thiếu nhi` | **`CHILDREN`** | Pedagogical nursery song genre | **HIGH** |
| `Nhạc Trịnh / Tác phẩm Trịnh Công Sơn` | **`NHAC_TRINH`** | Classical Vietnamese author genre | **HIGH** |
| `Rock Việt / Vietnamese Rock` | **`ROCK`** | Acoustic / electric band recording | **HIGH** |
| `R&B Việt / R&B Soul` | **`RB_SOUL`** | Contemporary rhythm & soul genre | **HIGH** |
| `Nhạc Cách Mạng / Nhạc Đỏ` | **`REVOLUTIONARY`** | Patriotic historical anthems | **HIGH** |
| `Rap Việt / Hip Hop` | **`RAP_HIPHOP`** | Vietnamese rhythm and poetry | **HIGH** |
| `Dance Việt / EDM Việt` | **`DANCE_EDM`** | Electronic dance genre | **HIGH** |
| `Nhạc Dân Ca / Quê Hương / Ca Trù / Chèo` | **`FOLK_TRADITIONAL`** | Traditional heritage folklore | **HIGH** |
| `Nhạc Tôn Giáo / Thánh Ca / Nhạc Phim (OST)` | **`OTHER`** | Verified positive out-of-taxonomy items | **HIGH** |
| `Cải Lương (Traditional Opera)` | **`UNMAPPED`** | Theatrical stage opera; distinct from song taxonomy | **UNMAPPED** |
| `Unknown / Unclassified` | **`UNMAPPED`** | Insufficient semantic ground truth | **UNMAPPED** |
