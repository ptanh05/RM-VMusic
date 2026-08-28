# Genre Taxonomy & Label Provenance Specification

This document establishes the Genre Taxonomy, Normalization Mapping, and Label Provenance Rules for the **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)** project.

---

## 1. Grounded Context & Principles

1. **Non-Novelty Stance**: This taxonomy does **not** claim to be a novel theoretical classification. Instead, it systematically normalizes real-world, highly inconsistent genre tags crawled from Vietnamese streaming platforms (primarily Zing MP3 via VietLyrics) into a well-defined set of standardized genre categories for machine learning research.
2. **Label Provenance Tracking**: Every sample records:
   - `source_genre`: Raw genre string from source platform (e.g. `nhạc trẻ`, `v-pop`, `nhạc trữ tình`, `rap việt`).
   - `genre`: Standardized normalized taxonomy class.
   - `label_source`: Provenance category (`original_source`, `deterministic_mapping`, `manual_annotation`, `unknown`).
   - `annotation_status`: Verification state (`cross_verified`, `normalized`, `needs_manual_annotation`).
3. **Ambiguity Handling**: Any tag that conflates genre with song origin, format, or performance style (e.g., `Nhạc Hoa Lời Việt`, `Acoustic`, `Remix`, `Cover`) is explicitly marked as `NEEDS_MANUAL_ANNOTATION` if the primary musical genre cannot be deterministically resolved.
4. **No Guessing / No Fabrication**: Missing genre labels remain `NULL` or are flagged as `NEEDS_MANUAL_ANNOTATION` with `label_source = "unknown"`.

---

## 2. Standard Normalized Genre Classes

| Normalized Genre Code | Canonical Name | Description & Vietnamese Musical Context |
|-----------------------|----------------|------------------------------------------|
| `POP_BALLAD` | **V-Pop / Ballad** | Nhạc trẻ đương đại, pop ballads, acoustic pop, teen pop. |
| `BOLERO_TRUTINH` | **Trữ tình / Bolero** | Nhạc vàng, Bolero Việt Nam, nhạc trữ tình quê hương chậm buồn. |
| `RAP_HIPHOP` | **Rap / Hip-Hop Việt** | Rap Việt, underground/overground hip-hop, trap. |
| `DANCE_EDM` | **Dance / Electronic / Remix** | Vinahouse, EDM, Electro Pop, House, Club Remixes. |
| `CHILDREN` | **Thiếu nhi** | Ca khúc thiếu nhi, bài hát học đường mầm non - tiểu học. |
| `FOLK_TRADITIONAL` | **Dân ca / Quê hương** | Dân ca ba miền (Bắc Bộ, Trung Bộ, Nam Bộ), cải lương, hát chầu văn, ca trù. |
| `REVOLUTIONARY` | **Tiền chiến / Cách mạng** | Nhạc đỏ, ca khúc cách mạng, hành khúc truyền thống. |
| `ROCK` | **Rock Việt** | Hard rock, alternative rock, indie rock, pop rock (Hiếm trong nguồn hiện tại). |
| `NHAC_TRINH` | **Nhạc Trịnh** | Dòng nhạc Trịnh Công Sơn mang phong cách triết lý, tự sự đặc trưng. |
| `RB_SOUL` | **R&B / Soul** | Contemporary R&B, Soul, Neo-Soul Việt. |
| `INSTRUMENTAL` | **Nhạc không lời / Hòa tấu** | Nhạc hòa tấu nhạc cụ, độc tấu, nhạc phim không lời. |

---

## 3. Explicit Source Genre Mapping & Provenance Rules

| Raw `source_genre` String | Normalized `genre` | `label_source` | `annotation_status` |
|---------------------------|--------------------|----------------|---------------------|
| `nhạc trẻ`, `v-pop`, `vpop`, `pop`, `teen pop`, `ballad` | `POP_BALLAD` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc trữ tình`, `bolero`, `nhạc vàng`, `trữ tình & bolero` | `BOLERO_TRUTINH` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `rap việt`, `rap`, `hip hop`, `rap / hip hop việt`, `hip-hop` | `RAP_HIPHOP` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `dance`, `edm`, `dance / edm`, `vinahouse`, `remix`, `dance việt`, `edm việt` | `DANCE_EDM` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc cách mạng`, `nhạc đỏ`, `tiền chiến`, `truyền thống cách mạng` | `REVOLUTIONARY` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc quê hương`, `dân ca`, `dân ca quê hương`, `quan họ`, `nhạc dân ca - quê hương` | `FOLK_TRADITIONAL` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc thiếu nhi`, `thiếu nhi` | `CHILDREN` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc trịnh`, `trịnh công sơn` | `NHAC_TRINH` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `r&b`, `r&b / soul`, `soul`, `r&b việt` | `RB_SOUL` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `rock việt`, `rock`, `hard rock` | `ROCK` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc không lời`, `không lời`, `instrumental`, `hòa tấu` | `INSTRUMENTAL` | `deterministic_mapping` | `normalized` / `cross_verified` |
| `nhạc hoa lời việt`, `cover`, `acoustic`, `unknown`, `null`, `nan` | `NEEDS_MANUAL_ANNOTATION` | `unknown` | `needs_manual_annotation` |

---
*Tài liệu thuộc khuôn khổ dự án RM-VMusic - Phase 4.*
