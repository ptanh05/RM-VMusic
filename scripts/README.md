# Script status

Only `python scripts/run_all.py <stage>` is the supported pipeline entry point.

- **ACTIVE:** `run_all.py`; `extract_features.py` is a compatibility alias for `prepare`.
- **RETIRED SAFETY SHIMS:** `extract_features_v4.py`, `build_dataset_v4_audio_expanded.py`, and hash extractors in `train_baseline.py`. They now refuse unsafe operations.
- **LEGACY:** historical V1/final/final12, baseline, evaluation, report, phase6–phase9 and V3 benchmark scripts. Their outputs are retained, but these scripts do not define the current method or reproducibility path.
- **EXPERIMENTAL:** V2/V4, external ingestion, integration, expansion, phase11–phase16, imbalance and discovery scripts. They are excluded from official-v1.
- **NETWORK/DATA CURATION:** collect, recover, ingest and discovery scripts may access networks or mutate historical datasets. The official pipeline never invokes them.

The complete per-file pre-change inventory and hashes are in `docs/CODEBASE_AUDIT.md`. Scripts were previously excluded by `.gitignore`; they are now visible to Git, but no files were staged or committed.
