# Official data pipeline

## Status and authority

The sole supported pipeline is `official-v1`, configured by `configs/official.yaml` and invoked through `scripts/run_all.py`. Dataset V3 is selected because the repository's Phase 16 decision ratifies it and because it contains the complete existing 12-class mapping without asserted audio records that lack linked assets.

V1/final, V2, V4, V4-expanded, final12 flat splits and `data/splits/legacy/` are retained. They are historical or experimental inputs and are not read by official-v1.

## Frozen source

- Metadata: `data/processed/final_12class_metadata_v3.csv`
- Rows / unique song IDs: 5,569 / 5,569
- SHA-256: `b933b94bac5c9796bfb1deecea12745ce89510d2b85c069c0f74388a5fba6f4a`
- Existing label order: `POP_BALLAD, BOLERO_TRUTINH, INSTRUMENTAL, RAP_HIPHOP, FOLK_TRADITIONAL, DANCE_EDM, REVOLUTIONARY, NHAC_TRINH, ROCK, RB_SOUL, OTHER, CHILDREN`
- Linked local assets observed on 2026-09-09: 4,117 lyrics, 902 covers, 0 audio
- Verified release-year rows: 770

The config checksum and row count must match before any stage runs. Unknown labels, duplicate IDs, blank IDs, or changed label order fail validation.

## Split rules

All splits are regenerated deterministically under `artifacts/official-v1/splits/`, seed 42. Existing files under `data/splits/` are neither trusted nor overwritten.

- IID: stratified 70/15/15.
- Artist-disjoint: normalized artist groups, stratified by each artist's primary genre. Normalization applies Unicode NFKC, case folding and whitespace collapsing. The same normalization verifies zero overlap.
- Temporal: only `year_status=verified`; train ≤2018, validation 2019–2020, test ≥2021.
- Label shift: dominant `POP_BALLAD` and remaining classes use fixed, seeded sampling fractions defined in `src/data/splits/label_shift.py`.
- Missing modality: an exact copy of the IID test rows; masks are changed only during evaluation.

Every scenario checks song-ID disjointness, row content against frozen metadata and expected coverage. Temporal coverage is the verified-year subset; other scenarios cover all 5,569 IDs.

Current deterministic row counts are:

| Scenario | Train | Validation | Test |
|---|---:|---:|---:|
| IID | 3,898 | 835 | 836 |
| Artist-disjoint | 3,931 | 842 | 796 |
| Temporal | 526 | 54 | 190 |
| Label shift | 3,944 | 588 | 1,037 |

## Feature extraction

A separate cache is produced for each scenario because TF-IDF is fitted only on that scenario's train song IDs. All 5,569 metadata rows remain in each cache so a split can index by ID without changing row identity.

- Lyrics, shape `[5569, 5000]`: UTF-8 physical text; train-only unigram/bigram TF-IDF; zero padding if the vocabulary has fewer than 5,000 terms.
- Cover, shape `[5569, 512]`: decoded RGB image resized to 128×128; 3×3 spatial channel histograms, RGB mean/std and gradient histogram.
- Audio, shape `[5569, 128]`: decoded WAV bytes, at most ten seconds, mixed to mono, resampled to 16 kHz, 128 log spectral bands. V3 has no linked audio, so its matrix is all-zero and mask is all-zero.
- Mask, shape `[5569]`: exactly 1 only after physical reading and valid nonzero feature extraction. Missing, unsupported, corrupt, silent or non-finite files remain zero with mask 0.

Each cache contains `song_ids.json`, `vectorizer.json`, arrays, asset hashes/statuses and `manifest.json`. Cache loading validates schema, extraction implementation hashes, dataset/config/split context, train IDs, dimensions, finite values, binary masks, missing rows, and every file checksum. Pickle is not used by official-v1.

## Dataset, model, evaluation

`FeatureTensorDataset` maps dataframe song IDs to cache indices in dataframe order. Missing IDs raise. Explicit `missing_policy=drop` filters the dataframe and labels together, warns, and exposes dropped IDs; the official pipeline uses the default error policy.

The six configured models share the same six-input interface: lyrics/cover/audio feature tensors and masks. UAD-Fusion uses positive learned uncertainty, exponential reliability and masked normalized weights. Training selects checkpoints using validation Macro-F1. Evaluation reads the test set only after training and records per-song probabilities, masks and UAD weights.

## Exclusions

Official-v1 excludes metadata-only audio URLs, generated numbered recordings, hash/random features, shared feature caches in `data/features/`, V4 data, network acquisition, pseudo labels, and all historical report values. Preparation never downloads assets and never trains.
