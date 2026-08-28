# RM-VMusic Phase 6: Audio Quality and Physical Coverage Audit Report

This report evaluates physical audio availability on disk versus metadata URL coverage across **5,416** trainable Vietnamese music tracks.

---

## 1. Executive Audio Audit Summary

- **Total Trainable Tracks**: **5,416**
- **Metadata Audio URL Coverage**: **5,401 / 5,416 (99.72%)**
- **Physical Audio File Coverage on Disk**: **0 / 5,416 (0.00%)**
- **Physical Recovery Gap**: **5,416 tracks**

---

## 2. Genre-by-Genre Audio Coverage Matrix

| Genre | Total ($N$) | Metadata URL Count (%) | Physical Audio Valid (%) | Physical Recovery Gap | Status |
|-------|-------------|------------------------|--------------------------|-----------------------|--------|
| `POP_BALLAD` | 3031 | 3023 (99.74%) | **0 (0.0%)** | **3031** | `PHYSICALLY_UNMATERIALIZED` |
| `BOLERO_TRUTINH` | 807 | 804 (99.63%) | **0 (0.0%)** | **807** | `PHYSICALLY_UNMATERIALIZED` |
| `INSTRUMENTAL` | 287 | 287 (100.0%) | **0 (0.0%)** | **287** | `PHYSICALLY_UNMATERIALIZED` |
| `RAP_HIPHOP` | 221 | 221 (100.0%) | **0 (0.0%)** | **221** | `PHYSICALLY_UNMATERIALIZED` |
| `FOLK_TRADITIONAL` | 200 | 197 (98.5%) | **0 (0.0%)** | **200** | `PHYSICALLY_UNMATERIALIZED` |
| `DANCE_EDM` | 193 | 193 (100.0%) | **0 (0.0%)** | **193** | `PHYSICALLY_UNMATERIALIZED` |
| `REVOLUTIONARY` | 170 | 170 (100.0%) | **0 (0.0%)** | **170** | `PHYSICALLY_UNMATERIALIZED` |
| `NHAC_TRINH` | 145 | 144 (99.31%) | **0 (0.0%)** | **145** | `PHYSICALLY_UNMATERIALIZED` |
| `ROCK` | 137 | 137 (100.0%) | **0 (0.0%)** | **137** | `PHYSICALLY_UNMATERIALIZED` |
| `RB_SOUL` | 132 | 132 (100.0%) | **0 (0.0%)** | **132** | `PHYSICALLY_UNMATERIALIZED` |
| `CHILDREN` | 93 | 93 (100.0%) | **0 (0.0%)** | **93** | `PHYSICALLY_UNMATERIALIZED` |

---

## 3. Technical Audit Inferences & Provenance Notes
1. **URL Expiration**: The 4,406 Zing MP3 streaming URLs (`a128-z3.zmdcdn.me`) require active time-based token authorization (`authen=exp=...&s=...`) which has expired since original metadata indexing.
2. **MusicBrainz Recordings**: 823 tracks link to MusicBrainz recording entities (`musicbrainz.org/recording/...`) which contain metadata catalog IDs rather than direct audio waveforms.
3. **Scientific Reporting Correction**: Previous reports quoting `99.72% audio coverage` reflected metadata URL presence. The physical audio coverage on disk is currently **0.00%**, requiring raw audio materialization or acoustic feature cache usage in future phases.
