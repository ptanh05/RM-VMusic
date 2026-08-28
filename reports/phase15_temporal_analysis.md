# RM-VMusic Phase 15: Deep Temporal Provenance & Archival Analysis
**Evaluation Date:** 2026-08-28

---

## 1. Temporal Field Provenance & Integrity

| Temporal Field Type | Definition | Verification Standard | Benchmark Action |
|---|---|---|---|
| **`song_creation_year`** | Historical composition date by the author (e.g. 1970 for Trịnh Công Sơn) | Archival musicology | Contextual reference only; not used for distribution shift |
| **`recording_year`** | Year master audio was captured in the studio | Studio recording session logs | Secondary metadata |
| **`album_release_year`** | Commercial publication date of physical/digital album | Official discography publication | **Verified Benchmark Release Year** |
| **`upload_year`** | Timestamp a user uploaded a file to YouTube / streaming | Platform upload timestamp | **REJECTED (Upload timestamp != Release Year)** |
| **`metadata_crawl_year`** | Date the web crawler scraped the metadata | Crawler execution timestamp | **REJECTED (Scrape date != Release Year)** |

---

## 2. Special Musicological Findings

### A. Nhạc Trịnh Historical Integrity
- *Trịnh Công Sơn* (1939–2001) composed and recorded his foundational discography during the 20th century.
- Modern indie covers uploaded after 2021 on streaming platforms cannot be legitimately classified as modern historical baseline tracks without distorting authorial genre semantics.
- Therefore, the presence of 95 verified tracks $\le 2018$ and 0 tracks $\ge 2021$ in our open catalog is a faithful reflection of music history.

### B. Children's Songs Archival Reality
- Children's nursery songs in Vietnamese public archives represent traditional pedagogy recordings from the 2000s (2004–2008). No verified post-2021 digital catalog exists with open redistribution rights.

---

## 3. Verified Temporal Benchmark Partition Summary
- **Verified Release Year Catalog:** **770 tracks (13.83%)**
  - Historical Train ($\le 2018$): **526 tracks**
  - Transition Val ($2019–2020$): **54 tracks**
  - Modern Test ($\ge 2021$): **190 tracks**
- **Active Temporal Test Space:** **10 / 12 classes**.
