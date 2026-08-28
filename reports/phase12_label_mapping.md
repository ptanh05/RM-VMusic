# RM-VMusic Phase 12: Cross-Dataset Semantic Label Mapping

This document specifies the exact mapping from external Vietnamese music genre taxonomies into the standard RM-VMusic 12-Class Taxonomy.

| External Taxonomy Label | Vietnamese Semantic Meaning | RM-VMusic 12 Target Class | Evidence & Grounding | Mapping Confidence |
|---|---|---|---|---|
| `nhạc trẻ`, `v-pop`, `pop` | Contemporary Vietnamese pop music | `POP_BALLAD` | Upstream metadata tag | **1.00 (Exact Match)** |
| `nhạc trữ tình`, `bolero`, `trữ tình & bolero` | Vintage lyrical romantic ballads | `BOLERO_TRUTINH` | Upstream metadata tag | **1.00 (Exact Match)** |
| `rap việt`, `hip hop`, `rap` | Vietnamese rap & hip hop | `RAP_HIPHOP` | Upstream metadata tag | **1.00 (Exact Match)** |
| `nhạc dân ca`, `quê hương`, `ca trù`, `chèo`, `hát xẩm` | Traditional Vietnamese folklore music | `FOLK_TRADITIONAL` | Cultural music classification | **0.95 (Semantic Match)** |
| `dance việt`, `edm việt`, `nhạc dance` | Electronic dance & house music | `DANCE_EDM` | Electronic genre taxonomy | **1.00 (Exact Match)** |
| `nhạc cách mạng`, `nhạc đỏ` | Patriotic & revolutionary anthems | `REVOLUTIONARY` | Genre taxonomy standard | **1.00 (Exact Match)** |
| `nhạc trịnh` | Discography of Trịnh Công Sơn | `NHAC_TRINH` | Author genre classification | **1.00 (Exact Match)** |
| `rock việt`, `rock`, `alternative` | Vietnamese rock band recordings | `ROCK` | Acoustic band taxonomy | **0.95 (Semantic Match)** |
| `r&b việt`, `r&b / soul`, `blues` | Contemporary R&B and Soul | `RB_SOUL` | Groove / R&B classification | **0.95 (Semantic Match)** |
| `nhạc thiếu nhi` | Children's nursery rhymes & songs | `CHILDREN` | Pedagogical music genre | **1.00 (Exact Match)** |
| `new age`, `nhạc không lời`, `guitar`, `world music` | Instrumental / ambient tracks | `INSTRUMENTAL` | Non-vocal arrangement | **0.95 (Semantic Match)** |
| `nhạc tôn giáo`, `nhạc đạo`, `nhạc phim (OST)`, `âu mỹ` | Sacred hymns, soundtracks, western | `OTHER` | Positive out-of-taxonomy evidence | **0.90 (Verified Evidence)** |
| `unknown`, `NaN`, `chưa phân loại` | Unlabelled or ambiguous | **QUARANTINE** | Insufficient evidence | **0.00 (REJECTED)** |
