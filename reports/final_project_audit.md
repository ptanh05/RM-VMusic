# RM-VMusic: Final End-to-End Project Audit & Integrity Verification
**Audit Date:** 2026-08-28  
**Project:** Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift  
**Standard:** Strict Scientific Ground Truth (No Pseudo-Features, Zero Fabricated Data)

---

## 1. Catalog & Physical Asset Verification

| Dimension | Physical Files on Disk | Validated Non-Corrupt | Dataset Coverage ($N=5,515$) | Scientific Modeling Status |
|---|---|---|---|---|
| **Audio Waveforms (`data/audio/`)** | **0 files** | **0 files** | **0.00%** | Physical audio absent due to copyright boundaries; strictly handled via **Zero-Masking** ($mask=0.0$) |
| **Album Cover Art (`data/covers/`)**| **1,445 files** | **1,445 files** | **16.36% (902 tracks)** | Validated JPEG/PNG images; decoded with Pillow into 512-dim visual spatial color moments |
| **Song Lyrics (`data/lyrics/`)** | **4,117 files** | **4,117 files** | **74.65% (4,117 tracks)**| Validated UTF-8 NFC text files; extracted into 5,000-dim TF-IDF n-gram vectors |
| **Verified Release Years** | **770 tracks** | **770 tracks** | **13.96%** | Verified range 1967–2026 (4,745 tracks excluded from temporal shift partition) |

---

## 2. 12-Class Taxonomy & Semantic Definition of `OTHER`

| Index | Genre Class | Sample Count | Percentage | Unique Artists | Semantic Category Scope |
|---|---|---|---|---|---|
| 0 | `POP_BALLAD` | 3,031 | 54.96% | 1,890 | Vietnamese Pop, Lyrical Ballads, Acoustic slow-tempo |
| 1 | `BOLERO_TRUTINH` | 807 | 14.63% | 501 | Bolero, Nostalgic Love Ballads, Rhumba |
| 2 | `INSTRUMENTAL` | 287 | 5.20% | 141 | Instrumental solos (guitar, piano, saxophone, traditional) |
| 3 | `RAP_HIPHOP` | 221 | 4.01% | 111 | Vietnamese Underground Rap, Hip-hop, Trap |
| 4 | `FOLK_TRADITIONAL` | 200 | 3.63% | 77 | Regional folk songs, Quan họ, Vọng cổ, Chèo |
| 5 | `DANCE_EDM` | 193 | 3.50% | 139 | Electronic Dance Music, Vinahouse, Club Remixes |
| 6 | `REVOLUTIONARY` | 170 | 3.08% | 31 | Red Music (*Nhạc Đỏ*), Patriotic military anthems |
| 7 | `NHAC_TRINH` | 145 | 2.63% | 23 | Trịnh Công Sơn philosophical compositions |
| 8 | `ROCK` | 137 | 2.48% | 20 | Vietnamese Alternative Rock, Hard Rock, Metal |
| 9 | `RB_SOUL` | 132 | 2.39% | 27 | Contemporary R&B, Neo-Soul, Groove |
| 10 | `OTHER` | 99 | 1.80% | 54 | Verified Out-of-Taxonomy (Sacred Hymns, Film OSTs, Country) |
| 11 | `CHILDREN` | 93 | 1.69% | 41 | Nursery rhymes, Children songs |
| **Total** | **12 Classes** | **5,515** | **100.00%** | **2,746** | **Max Imbalance: 32.59x | Gini Index: 0.6102** |

---

## 3. Data Integrity & Leakage Verification

- **Duplicate `song_id`:** 0
- **Duplicate `(title, artist)`:** 0
- **Duplicate `source_id`:** 0
- **Artist Disjoint Leakage:**
  $$\text{Train Artists } (N=1,908) \cap \text{Val Artists } (N=428) = 0$$
  $$\text{Train Artists } (N=1,908) \cap \text{Test Artists } (N=411) = 0$$
  $$\text{Val Artists } (N=428) \cap \text{Test Artists } (N=411) = 0$$
- **Pseudo-Feature Elimination:** 100% verified. Deterministic SHA-256 hash embeddings have been permanently deleted and replaced by genuine multimodal feature encoders with explicit zero-masking for missing modalities.
