# RM-VMusic

RM-VMusic is a research codebase for 12-class Vietnamese music genre classification using lyrics, cover art, and optional physical audio. The supported software pipeline is **official-v1**. It is frozen to Dataset V3 (5,569 unique songs) and writes new artifacts only under `artifacts/official-v1/`.

The current V3 metadata links 4,117 lyrics files, 902 cover files, and **no audio files**. The 147 audio files elsewhere in the workspace belong to experimental datasets and are excluded from official-v1. Audio inputs for V3 are zero vectors with mask 0. They provide no acoustic evidence.

## Supported pipeline

```text
final_12class_metadata_v3.csv (frozen SHA-256)
  -> deterministic scenario splits
  -> physical asset inspection
  -> scenario-specific TF-IDF + visual/audio extraction
  -> versioned checksummed cache
  -> aligned FeatureTensorDataset
  -> baselines / UAD-Fusion
  -> predictions, metrics, calibration and missingness evaluation
```

`scripts/run_all.py` is the only supported entry point:

```bash
python scripts/run_all.py validate
python scripts/run_all.py smoke
python scripts/run_all.py prepare
python scripts/run_all.py train --run-id first_clean_run
python scripts/run_all.py evaluate --run-id first_clean_run
```

The default command, `python scripts/run_all.py`, performs read-only validation. `smoke` uses a tiny real subset and random model initialization for structural checks; it does not fit a model and its scores are not research results. `prepare` rebuilds scenario-specific feature caches without training. Training and evaluation are explicit separate stages.

Historical scripts and results remain in the repository. Their status is documented in [scripts/README.md](scripts/README.md) and they are excluded from official-v1.

## Installation

Python 3.12 is the verified environment. CPU is the documented default.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements.txt` pins the directly imported runtime packages tested for this cleanup. The verified local PyTorch build was 2.5.1+cu121, but official-v1 defaults to CPU and does not require CUDA. `torchvision` is not used.

## Features and missing modalities

- Lyrics: 5,000-dimensional TF-IDF, fit only on the train partition of each scenario. Smaller vocabularies are zero-padded to 5,000 columns.
- Cover: 512-dimensional descriptor from decoded JPEG pixels: 3×3 spatial RGB histograms, global RGB moments, and gradient histogram.
- Audio: 128-dimensional log spectral bands from successfully decoded local WAV bytes, resampled to mono 16 kHz and limited to the first 10 seconds.
- Missing/unsupported/invalid physical files: zero feature and binary mask 0. A URL, metadata availability flag, song ID, genre, or random number never constitutes a valid feature.

Official V3 has 0% linked audio coverage. MP3 is currently unsupported by the extractor; unsupported files remain missing instead of being marked valid.

## UAD-Fusion implementation

For each modality embedding `z_m`, the model computes

```text
u_m = softplus(g_m(z_m))
r_m = exp(-u_m) * mask_m
alpha_m = softmax(-u_m over available modalities)
z_fused = sum(alpha_m * z_m)
```

Unavailable modalities receive exactly zero reliability and fusion weight. A sample with no evidence has a zero fused embedding and an explicit `has_evidence=False` flag. The model exposes uncertainty, reliability, fusion weights, active masks, and embeddings for analysis. These learned scores rank modality reliability; they are not independently calibrated statistical variances.

UAD architecture version 1 intentionally rejects old checkpoints because the previous sigmoid-gate semantics were different.

## Evaluation

Metrics use the fixed 12-class label space for every split: accuracy, 12-class Macro-F1, weighted F1, per-class precision/recall/F1/support, ECE, multiclass Brier score, NLL, and confusion matrix. `balanced_accuracy` averages recall only over classes supported by the evaluated split and is reported together with `supported_class_count`.

The temporal split contains only verified years and does not contain every class in every partition. Its results therefore need that caveat. The overall dataset remains strongly imbalanced.

## Historical results

Files under `reports/` and `outputs/` predate official-v1. They were preserved unchanged. Their numeric claims have not been reproduced by the cleaned pipeline, and they must not be presented as results of the corrected UAD implementation until a new explicitly named run is trained and evaluated.

See [code audit](docs/CODEBASE_AUDIT.md), [data pipeline](docs/DATA_PIPELINE.md), [reproducibility](docs/REPRODUCIBILITY.md), and [implementation status](docs/IMPLEMENTATION_STATUS.md).
