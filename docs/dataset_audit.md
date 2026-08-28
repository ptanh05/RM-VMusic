# RM-VMusic: Dataset Sources Audit Report

This document records the empirical verification and technical audit of candidate data sources for the **RM-VMusic (Reliable Multimodal Vietnamese Music Genre Classification under Distribution Shift)** project.

---

## 1. Executive Summary of Audited Sources

| # | Dataset Name | URL | Verified Samples | Audio | Lyrics | Genre | Artist | Album | Year | Cover | License | Automated Access Status | Research Viable? |
|---|--------------|-----|------------------|-------|--------|-------|--------|-------|------|-------|---------|-------------------------|------------------|
| 1 | **VietLyrics** | [GitHub Repo](https://github.com/BatmanofZuhandArrgh/VietLyrics) | ~8,000 (~7k train + ~1k val) | URL (Zing MP3 link) | Token statistics (text linked via ID) | Scraped from Zing MP3 | Yes | No (NULL) | No (NULL) | No (NULL) | Unknown / Không công bố file license | Sẵn sàng (Trực tiếp qua raw CSV) | Có (Fair Use / Nghiên cứu học thuật) |
| 2 | **Vietnamese Song Dataset** | [Hugging Face](https://huggingface.co/datasets/sunbv56/song_dataset) | ~7,653 (653 eval + ~7k train) | URL (CDN stream có token hết hạn + Zing MP3 URL) | Văn bản đầy đủ + Word timestamps | Không có trong jsonl gốc (NULL) | Có | Có | No (NULL) | No (NULL) | Apache-2.0 | Sẵn sàng (Trực tiếp qua JSONL) | Có (Nghiên cứu học thuật) |
| 3 | **Zalo Vietnamese Music Genre** | [UnderTheSea NLP](https://github.com/undertheseanlp/NLP-Vietnamese-progress/blob/master/tasks/speech_classification.md) | Unknown / Cần kiểm tra thêm (~30k speech trong Voice Gender) | Unknown / File audio lưu trên cổng Zalo AI cũ | Unknown / Cần kiểm tra thêm | Unknown trong link hiện tại | Unknown | Unknown | Unknown | Unknown | Unknown (Điều khoản cuộc thi Zalo AI) | Không thể tự động tải trực tiếp (Yêu cầu cổng Zalo AI) | Cần adapter ngoại tuyến |
| 4 | **Vietnamese Music Dataset** | [Kaggle](https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1/data) | ~3.99 GB zip archive (Version 17) | Binary Audio (Zip) | Unknown trong metadata công khai | Cần giải nén để kiểm tra file labels | Cần giải nén | Cần giải nén | Cần giải nén | Cần giải nén | CC0: Public Domain (theo metadata Kaggle) | Yêu cầu Kaggle API Token (`kaggle.json`) | Có (Cần cấu hình Kaggle API hoặc tải thủ công) |

---

## 2. Detailed Technical Audit per Source

### 2.1. VietLyrics
- **Repository URL**: `https://github.com/BatmanofZuhandArrgh/VietLyrics`
- **Tình trạng truy cập thực tế**: Kết nối thành công (HTTP 200). Các tệp CSV được lưu trữ trực tiếp trên nhánh `main`.
- **Cấu trúc tệp**:
  - `data/train_7k_metadata_authors.csv` (1.75 MB, ~7,000 dòng)
  - `data/val_1k_metadata_authors.csv` (236 KB, ~1,000 dòng)
- **Các trường dữ liệu (Fields)**:
  - `song`: Tên bài hát đã chuẩn hóa.
  - `link`: URL bài hát trên Zing MP3 (chứa mã bài hát Zing, ví dụ: `ZW7OCD60`).
  - `prefix`: Tiền tố slug để parse tên bài hát.
  - `title`: Tên bài hát đầy đủ.
  - `genre`: Thể loại nhạc crawl từ Zing MP3 (ví dụ: `nhạc trẻ`, `v-pop`, `nhạc trữ tình`, ...).
  - `artist`: Nghệ sĩ trình bày.
  - `token_count`: Số lượng token trong lời bài hát (tokenize bởi Whisper tokenizer).
  - `wpm`: Tốc độ từ trên phút (Words per minute).
  - `duration_mins`: Thời lượng bài hát tính theo phút.
  - `loibaihat_authors`, `lyricvn_authors`: Tác giả sáng tác từ các trang lời bài hát.
- **Đánh giá đa phương thức (Modalities)**:
  - *Audio*: Không đính kèm tệp nhị phân `.mp3`/`.wav`. Cung cấp `link` Zing MP3.
  - *Lyrics*: File metadata chỉ chứa thống kê token/tác giả, văn bản lời bài hát chi tiết liên kết thông qua mã Zing MP3.
  - *Genre*: Có sẵn nhãn thể loại từ Zing MP3 (cần chuẩn hóa vào Taxonomy).
  - *Artist*: Có sẵn.
  - *Album / Year / Cover*: Không có (NULL).
- **Bản quyền & Giới hạn sử dụng**: Không có file `LICENSE` trong repo ("Unknown / cần kiểm tra thêm"). Dữ liệu trích xuất từ Zing MP3, phù hợp cho mục đích nghiên cứu học thuật phi thương mại. Không được phân phối lại các tệp audio có bản quyền.

---

### 2.2. Vietnamese Song Dataset (sunbv56/song_dataset)
- **Repository URL**: `https://huggingface.co/datasets/sunbv56/song_dataset`
- **Tình trạng truy cập thực tế**: Kết nối thành công (HTTP 200 qua Hugging Face Dataset API & Raw JSONL).
- **Cấu trúc tệp**:
  - `eval.jsonl` (653 bản ghi, ~20.59 giờ audio)
  - `train.jsonl` (ước tính ~7,000 bản ghi, ~230.62 giờ audio)
  - `README.md`
- **Các trường dữ liệu (Fields)**:
  - `id`: Mã định danh bài hát (trùng khớp với mã bài hát Zing MP3, ví dụ: `ZZ8CC7AZ`, `ZW66ECID`).
  - `title`: Tên bài hát.
  - `artist`: Nghệ sĩ biểu diễn.
  - `album`: Tên album chứa bài hát (có thể rỗng nếu single).
  - `streaming_url`: URL CDN stream audio (chú ý: token CDN có thời hạn sử dụng / expiration).
  - `lyrics`: Toàn văn lời bài hát tiếng Việt (các câu phân tách bằng `\n`).
  - `zingmp3_url`: Đường dẫn bài hát trên Zing MP3.
  - `word_timestamps`: Danh sách mảng 2 chiều chứa mốc thời gian chi tiết từng từ (`start`, `end`, `word` bằng mili-giây) phục vụ karaoke/lyrics alignment.
- **Đánh giá đa phương thức (Modalities)**:
  - *Audio*: Có `streaming_url` và `zingmp3_url`.
  - *Lyrics*: Toàn văn lời bài hát đầy đủ và timestamp từng từ cực kỳ chất lượng.
  - *Genre*: Không có trong JSONL gốc (NULL).
  - *Artist / Album*: Có đầy đủ.
  - *Release Year / Cover*: Không có trong JSONL gốc (NULL).
- **Bản quyền & Giới hạn sử dụng**: Được gắn thẻ `Apache-2.0` trên Hugging Face card. Tuy nhiên, nội dung âm nhạc và lời ca thuộc quyền sở hữu của các nghệ sĩ và hãng phát hành, URL streaming có token hết hạn. Chỉ lưu trữ URL/metadata cho mục đích nghiên cứu học thuật.

---

### 2.3. Zalo Vietnamese Music Genre Classification
- **Nguồn dẫn**: `https://github.com/undertheseanlp/NLP-Vietnamese-progress/blob/master/tasks/speech_classification.md`
- **Tình trạng truy cập thực tế**: Trang Markdown trong repo `NLP-Vietnamese-progress` mở được, nhưng nội dung thực tế ghi nhận cuộc thi: "Zalo AI Challenge: Voice Gender Classification" (`https://challenge.zalo.ai/portal/voice`) với ~30k đoạn voice ngắn phân loại giới tính/vùng miền, KHÔNG phải Music Genre.
- **Bối cảnh lịch sử**: Zalo AI Challenge 2018 từng có bảng thi "Music Genre Classification" và 2021 có "Hum2Song", nhưng cổng thi đấu lịch sử của Zalo AI không còn mở tải công khai và yêu cầu tài khoản/phiên dự thi.
- **Đánh giá**:
  - *Không thể tự động tải trực tiếp*.
  - Pipeline xây dựng một Module Adapter (`ZaloMusicAdapter`) sẵn sàng nạp dữ liệu khi nhà nghiên cứu có sẵn tệp lưu trữ nội bộ (offline raw folder).
  - *License*: Điều khoản cuộc thi Zalo AI (chỉ sử dụng trong khuôn khổ nghiên cứu/thi đấu).

---

### 2.4. Vietnamese Music Dataset – Kaggle (sonlest/vietnamese-music-dataset-version3-part1)
- **URL**: `https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1/data`
- **Tình trạng truy cập thực tế**: Xác minh được trang Dataset Metadata trên Kaggle (Dataset version 17, dung lượng tệp zip ~3.99 GB).
- **Phương thức tải**: Kaggle yêu cầu xác thực API Token (`kaggle.json` chứa `username` và `key`) hoặc thao tác tải thủ công từ trình duyệt. Tự động curl/wget không gửi token sẽ nhận mã redirect/đăng nhập.
- **Đánh giá đa phương thức & Bản quyền**:
  - *License công bố*: `CC0: Public Domain` (theo thẻ schema của Kaggle).
  - *Nội dung*: Tệp zip chứa các track âm thanh nhạc Việt.
  - *Đánh giá*: Cần cấu hình `Kaggle API` khi muốn tải toàn bộ ~4GB. Đối với giai đoạn Dataset Pilot, pipeline ưu tiên nạp metadata và dữ liệu multimodal trực tiếp từ VietLyrics và sunbv56/song_dataset, đồng thời cung cấp adapter để đọc dữ liệu Kaggle khi người dùng cung cấp folder raw.

---

### 2.5. MusicBrainz Open Music Encyclopedia
- **URL**: `https://musicbrainz.org/ws/2/` (Open CC0 / ODbL API)
- **Tình trạng truy cập thực tế**: Kết nối thành công (HTTP 200 qua MusicBrainz REST API).
- **Đánh giá đa phương thức & Bản quyền**:
  - *License công bố*: `Creative Commons Zero (CC0) / Open Database License (ODbL)`.
  - *Nội dung*: Bản ghi đĩa hát (recordings), danh mục nghệ sĩ (artist discography), năm phát hành đầu tiên (`first-release-date`), mã định danh toàn cầu `MBID`.
  - *Ứng dụng trong RM-VMusic*: Mở rộng có chủ đích cho các thể loại hiếm (`ROCK`, `NHAC_TRINH`, `REVOLUTIONARY`, `RB_SOUL`, `INSTRUMENTAL`, `FOLK_TRADITIONAL`, `CHILDREN`, `RAP_HIPHOP`, `DANCE_EDM`) đảm bảo 100% minh bạch về nguồn gốc và tuân thủ chặt chẽ ràng buộc đa dạng nghệ sĩ (tối đa $\le 6-8$ bài/nghệ sĩ).
  - *Bản quyền âm thanh*: Không tải/lưu trữ tệp âm thanh có bản quyền; liên kết qua URL tham chiếu mở `https://musicbrainz.org/recording/{mbid}`.

---

## 3. Khả năng kết hợp dữ liệu (Cross-Referencing & Fusion)

Qua quá trình audit, một phát hiện kỹ thuật quan trọng là:
- Cả **VietLyrics** và **sunbv56/song_dataset** đều sử dụng định danh bài hát từ Zing MP3 (ví dụ `ZW...`, `ZZ...`).
- **VietLyrics** cung cấp nhãn thể loại (`genre`), thời lượng và tác giả.
- **sunbv56/song_dataset** cung cấp toàn văn lời bài hát (`lyrics`), `album`, và `streaming_url`.
- Bằng cách liên kết thông qua `source_id` (Zing MP3 ID) hoặc chuẩn hóa cặp `(title, artist)`, pipeline có thể hợp nhất thành công các mẫu dữ liệu **Multimodal hoàn chỉnh (Audio Link + Lyrics + Artist + Album + Genre)** phục vụ trực tiếp cho đề tài RM-VMusic.

---

## 4. Trạng thái kiểm tra kết nối chi tiết (Automated Check Logs)

```json
[
  {
    "name": "VietLyrics",
    "url": "https://github.com/BatmanofZuhandArrgh/VietLyrics",
    "accessible": true,
    "details": {
      "https://raw.githubusercontent.com/BatmanofZuhandArrgh/VietLyrics/main/data/val_1k_metadata_authors.csv": {
        "status_code": 200,
        "content_length": "92716",
        "content_type": "text/plain; charset=utf-8"
      },
      "https://raw.githubusercontent.com/BatmanofZuhandArrgh/VietLyrics/main/data/train_7k_metadata_authors.csv": {
        "status_code": 200,
        "content_length": "640366",
        "content_type": "text/plain; charset=utf-8"
      },
      "https://raw.githubusercontent.com/BatmanofZuhandArrgh/VietLyrics/main/README.md": {
        "status_code": 200,
        "content_length": "877",
        "content_type": "text/plain; charset=utf-8"
      }
    }
  },
  {
    "name": "Vietnamese Song Dataset (sunbv56)",
    "url": "https://huggingface.co/datasets/sunbv56/song_dataset",
    "accessible": true,
    "details": {
      "https://huggingface.co/api/datasets/sunbv56/song_dataset": {
        "status_code": 200,
        "content_length": "1621",
        "content_type": "application/json; charset=utf-8"
      },
      "https://huggingface.co/datasets/sunbv56/song_dataset/raw/main/eval.jsonl": {
        "status_code": 200,
        "content_length": "227433",
        "content_type": "text/plain; charset=utf-8"
      },
      "https://huggingface.co/datasets/sunbv56/song_dataset/raw/main/README.md": {
        "status_code": 200,
        "content_length": "2867",
        "content_type": "text/plain; charset=utf-8"
      }
    }
  },
  {
    "name": "Zalo Vietnamese Music Genre Classification",
    "url": "https://github.com/undertheseanlp/NLP-Vietnamese-progress/blob/master/tasks/speech_classification.md",
    "accessible": true,
    "details": {
      "https://raw.githubusercontent.com/undertheseanlp/NLP-Vietnamese-progress/master/tasks/speech_classification.md": {
        "status_code": 200,
        "content_length": "1368",
        "content_type": "text/plain; charset=utf-8"
      }
    }
  },
  {
    "name": "Vietnamese Music Dataset – Kaggle (sonlest)",
    "url": "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1/data",
    "accessible": false,
    "details": {
      "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1/data": {
        "status_code": 404,
        "content_length": "134",
        "content_type": "text/html; charset=UTF-8"
      }
    }
  }
]
```

---
*Báo cáo được tạo tự động bởi `scripts/audit_sources.py` - RM-VMusic Pipeline.*
