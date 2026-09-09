# Reproducibility

## Verified environment

The cleanup and offline validation were executed on Windows, Python 3.12.10, PyTorch 2.5.1+cu121 and CUDA 12.1 availability. CPU is the official default. `requirements.txt` pins all direct runtime imports; `requirements-dev.txt` adds pytest. No `torchvision` dependency exists.

Install in a fresh environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
```

PyTorch wheels are platform-specific. If the exact CUDA-tagged wheel is required, install the matching official PyTorch wheel before the remaining requirements. The documented pipeline does not require CUDA.

## Frozen inputs and generated outputs

The source dataset, row count and SHA-256 are in `configs/official.yaml`. Physical source assets remain under `data/lyrics/`, `data/covers/`, and `data/audio/`. Preparation records the SHA-256 of every asset linked by the official metadata.

All new outputs are isolated below `artifacts/official-v1/`:

```text
metadata.csv
manifest.json
splits/<scenario>/{train,val,test}.csv
features/<scenario>/{arrays, song_ids.json, vectorizer.json, manifest.json}
runs/<run-id>/{config.json, environment.json, checkpoints, histories}
runs/<run-id>/evaluation/{predictions, metrics, summary.csv}
```

Existing `data/features/`, `outputs/`, and `reports/` are historical and are not consumed or overwritten.

## Commands

Read-only source and split validation:

```bash
python scripts/run_all.py validate
```

Full offline test suite and tiny real-data forward smoke:

```bash
python -m pytest -q
python scripts/run_all.py smoke
```

Prepare all deterministic splits and scenario-specific features without training:

```bash
python scripts/run_all.py prepare
python scripts/run_all.py validate --require-cache
```

Future explicit training and held-out evaluation:

```bash
python scripts/run_all.py train --run-id first_clean_run
python scripts/run_all.py evaluate --run-id first_clean_run
```

A run ID is required and cannot overwrite an existing run directory.

## Determinism and identity

The split seed is 42. Training seeds are 42, 123, 2024, 3407, and 7777. The helper seeds Python, NumPy, CPU/CUDA PyTorch, enables deterministic algorithms, disables cuDNN benchmarking, configures the cuBLAS workspace, and uses a seeded DataLoader generator. CPU thread count is fixed to one by config.

Exact cross-platform floating-point equality is not guaranteed across PyTorch/BLAS/CUDA implementations. Each run stores package/platform metadata, config digest, cache-manifest digest and checkpoint checksums so differences are observable.

TF-IDF is fitted separately for each train scenario. The cache refuses mismatched train IDs, source bytes, implementation, labels or metadata. The split generator fails if normalized artists overlap in the artist-disjoint scenario. Metric computation fixes the class denominator at 12.

## Git

Before this cleanup, `.gitignore` excluded all of `scripts/` and any `test_*.py`. These exclusions were removed. The working tree now exposes scripts, tests, config, docs and dependency manifests for review. This task does not stage, commit, push or change remotes; reproducibility from a clone begins only after the maintainer reviews and commits these visible files.

## Hardware assumptions

Validation, preparation, smoke and tests run on CPU. The full configured experiment is CPU-compatible but computationally expensive: four scenarios × six models × five seeds. CUDA may be selected explicitly in config only when available; there is no silent fallback. Memory must accommodate a dense 5,569×5,000 float32 lyrics matrix during extraction (about 106 MiB before temporary transformations).
