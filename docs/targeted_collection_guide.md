# RM-VMusic: Targeted Collection Strategy for Deficient Genres

This document details the targeted data acquisition framework designed to resolve severe class imbalances in **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)** without artificial oversampling or fake data.

---

## 1. Principles of Targeted Data Collection

1. **Empirical Grounding**: Data must be acquired from verified authentic Vietnamese musical recordings with real acoustic signals and verified lyrics.
2. **Artist Diversity Constraint**: To prevent the model from learning artist-specific shortcuts rather than acoustic/lyrical genre representations, a strict ceiling ($\max \le 8-10$ tracks per artist) is enforced for each genre class.
3. **No Synthetic / Oversampled Data**: Training sets must never duplicate or hallucinate records.

---

## 2. Targeted Genre Profiles & Collection Targets

| Genre Code | Canonical Name | Current Count | Target Sample Size | Max Tracks / Artist | Priority Level |
|------------|----------------|---------------|--------------------|---------------------|----------------|
| `ROCK` | **Rock Việt** | 7 | **200** | 8 | URGENT (Severely Deficient) |
| `RB_SOUL` | **R&B / Soul Việt** | 15 | **200** | 8 | URGENT (Severely Deficient) |
| `NHAC_TRINH` | **Nhạc Trịnh** | 10 | **150** | 10 | URGENT (Severely Deficient) |
| `INSTRUMENTAL` | **Nhạc không lời / Hòa tấu** | 10 | **150** | 6 | URGENT (Severely Deficient) |
| `REVOLUTIONARY` | **Tiền chiến / Cách mạng** | 19 | **150** | 6 | URGENT (Severely Deficient) |
| `FOLK_TRADITIONAL` | **Dân ca / Quê hương** | 73 | **300** | 8 | HIGH |
| `CHILDREN` | **Thiếu nhi** | 64 | **250** | 8 | HIGH |

---

## 3. Detailed Acquisition Profiles & Discography Sources

### 3.1. Rock Việt (`ROCK`)
- **Target Count**: 200 (Current: 7)
- **Max Tracks per Artist**: 8
- **Representative Artists**: Bức Tường, Microwave, Ngũ Cung, Trần Lập, Unlimited, Quái Vật Tí Hon, Parasite, Cát, I-Tễu, Gạt Tàn Đầy
- **Verified Query Sources**:
  - MusicBrainz: tag:vietnamese_rock OR artist:"Bức Tường"
  - VietLyrics: query keyword 'rock' in title or artist
  - Open Audio Archives / Public YouTube Creative Commons tracks

### 3.2. R&B / Soul Việt (`RB_SOUL`)
- **Target Count**: 200 (Current: 15)
- **Max Tracks per Artist**: 8
- **Representative Artists**: JustaTee, Touliver, Vũ., Marzuz, Mỹ Anh, Orange, Wren Evans, Phùng Khánh Linh, Thịnh Suy, Kimmese
- **Verified Query Sources**:
  - MusicBrainz: genre:r&b AND country:VN
  - VietLyrics: query 'r&b' or 'soul'
  - SpaceSpeakers public discographies

### 3.3. Nhạc Trịnh (`NHAC_TRINH`)
- **Target Count**: 150 (Current: 10)
- **Max Tracks per Artist**: 10
- **Representative Artists**: Khánh Ly, Trịnh Công Sơn, Trịnh Vĩnh Trinh, Lô Thủy, Giang Trang, Hồng Nhung, Cẩm Vân, Quang Dũng
- **Verified Query Sources**:
  - Trịnh Công Sơn catalog registries
  - VietLyrics: author:"Trịnh Công Sơn" OR title matches Trịnh classics
  - MusicBrainz: composer:"Trịnh Công Sơn"

### 3.4. Nhạc không lời / Hòa tấu (`INSTRUMENTAL`)
- **Target Count**: 150 (Current: 10)
- **Max Tracks per Artist**: 6
- **Representative Artists**: Đặng Thái Sơn, Võ Vân Ánh, Hải Phượng, Nguyễn Lê, Hoàng Tuấn, Kim Sinh, Hòa tấu đàn tranh, Hòa tấu đàn bầu
- **Verified Query Sources**:
  - VietLyrics 'new age / world music' instrumental queue
  - Traditional Instrument Archives (Dan Tranh, Dan Bau, Sao Truc)
  - MusicBrainz: instrument:solo

### 3.5. Tiền chiến / Cách mạng (`REVOLUTIONARY`)
- **Target Count**: 150 (Current: 19)
- **Max Tracks per Artist**: 6
- **Representative Artists**: Văn Cao, Phan Huỳnh Điểu, Hoàng Vân, Đỗ Nhuận, Trần Tiến, Đoàn Chuẩn, Quang Thọ, Trọng Tấn, Đăng Dương, Việt Hoàn
- **Verified Query Sources**:
  - Red Music public cultural archives
  - VietLyrics: genre 'nhạc cách mạng' or 'nhạc đỏ'
  - MusicBrainz: composer:"Văn Cao" OR composer:"Hoàng Vân"

### 3.6. Dân ca / Quê hương (`FOLK_TRADITIONAL`)
- **Target Count**: 300 (Current: 73)
- **Max Tracks per Artist**: 8
- **Representative Artists**: Thúy Hường, Thu Hiền, Thanh Hoa, Ánh Tuyết, Vân Khánh, Hương Lan, Quang Linh, Phi Nhung, Lệ Thủy, Minh Vương
- **Verified Query Sources**:
  - VietLyrics 'nhạc dân ca - quê hương' and 'cải lương'
  - Quan họ Bắc Ninh & Đờn ca tài tử open archives

### 3.7. Thiếu nhi (`CHILDREN`)
- **Target Count**: 250 (Current: 64)
- **Max Tracks per Artist**: 8
- **Representative Artists**: Xuân Mai, Bé Bào Ngư, Bé Trang Thư, Đội văn nghệ Tuổi Thơ, Bé Nhật Lan Vy, Bé Ben, Thiếu nhi Ba Đình
- **Verified Query Sources**:
  - VietLyrics 'nhạc thiếu nhi'
  - Tuổi Thơ music catalog registries

---
*Tài liệu thuộc khuôn khổ dự án RM-VMusic - Task 5 Targeted Data Collection.*
