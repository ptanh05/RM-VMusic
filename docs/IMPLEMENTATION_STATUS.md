# Implementation status

Validated on 2026-09-09 without full training and without network access.

| Component | Status | Notes |
|---|---|---|
| Dataset | PASS | V3 frozen at 5,569 unique IDs; unknown labels and silent ID drops rejected. |
| Split | PASS | Four deterministic scenarios; normalized artist disjointness and content/coverage checks pass. |
| Lyrics | PASS | Physical UTF-8 files; TF-IDF fitted per scenario train IDs; fixed 5,000 columns. |
| Cover | PASS | 902 files linked by V3; pixel-derived 512-dimensional features; invalid decode masks missing. |
| Audio | PASS | No V3 audio path; zero vectors/mask 0. WAV extractor is byte-derived and tested; MP3 is explicitly unsupported. |
| UAD-Fusion | PASS | Softplus uncertainty, exp(-u), masked normalized weights, stable no-evidence path and inspectable outputs. |
| Baselines | PASS | Six-input API for single, early and late fusion; forward smoke passes. |
| Evaluation | PASS | Fixed 12-class metrics, ECE/Brier/NLL, calibration bins, per-song prediction export. |
| Calibration | PASS | Implementation sanity tested; no corrected-model empirical calibration claim exists yet. |
| Tests | PASS | 65 offline tests pass; real-data forward smoke passes for all six configured models. |
| Reproducibility | PASS WITH ACTION | Versioned config/cache/run manifests and hashes implemented. Maintainer must commit visible scripts/tests/docs. |
| Documentation | PASS | Live README, pipeline, method, audit and reproducibility docs match official-v1. Historical reports remain labeled historical. |

## Remaining research risks

Dataset labels and source licenses were not independently re-annotated or externally verified. Class imbalance remains severe. The temporal evaluation has only 770 total rows and omits classes from some partitions. V3 has no acoustic evidence, so official-v1 can study lyrics/cover fallback and missing audio but cannot support claims about audio classification or trimodal gains. Cover/lyrics duplication across artists has not yet been assessed with a research-grade perceptual/text near-duplicate method.

Historical metrics were produced by older code. They remain intact under `reports/` and `outputs/`, but are not validation evidence for UAD architecture version 1.

## Final validation evidence

- Unit/sanity suite: `65 passed`.
- Real-data smoke: `PASS`, 24 source rows, 12 evaluation rows, all six configured models, no training.
- Full preparation and `validate --require-cache`: `PASS`; 51 generated files, 528,056,829 bytes.
- Scenario vocabulary sizes: IID 5,000; artist-disjoint 5,000; label-shift 5,000; temporal 2,562 (zero-padded to 5,000 feature columns).
- Every official audio cache contains zero nonzero values and zero positive masks, matching V3's zero linked-audio coverage.
- Historical protected trees before/after: 6,090 files, 1,068,022,247 bytes, aggregate SHA-256 `8e23c6bdc9132286b27473b78d79baafda5e67cc57dfb845375643391f6b7f92`.
- Syntax/import compilation completed. Python 3.12 emits invalid-escape warnings in retained legacy report strings; official-v1 modules do not rely on them.

## Training gate

Software preparation, dataset alignment, split isolation, model forward behavior, metrics and offline reproducibility checks are sufficient to prepare a new run. The full scenario caches were generated and passed `validate --require-cache` on 2026-09-09. Training should begin only after a maintainer accepts V3 labels/licenses and the stated no-audio scope. Results must use a new run ID and must not reuse old checkpoints.
