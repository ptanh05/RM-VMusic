# RM-VMusic: Reproducibility & Scientific Replication Guide

This guide specifies the exact, step-by-step procedures to replicate the entire RM-VMusic experimental benchmark from scratch.

---

## 1. Environment & Dependencies

- **OS:** Windows 10/11 or Ubuntu Linux 22.04+
- **Python Version:** 3.10, 3.11, or 3.12 (UTF-8 encoding enabled)
- **PyTorch Version:** PyTorch 2.5.1+ (CUDA or CPU compatible)

### Installation
```bash
# Clone and enter repository
git clone https://github.com/ptanh05/RM-VMusic.git
cd RM-VMusic

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. One-Click Master Pipeline Replication

To execute all 9 pipeline stages sequentially (Materialization $\to$ Dataset Build $\to$ 5 Splits $\to$ Leakage Audit $\to$ Feature Extraction $\to$ Baselines & UAD-Fusion Training $\to$ 12 Publication Figures $\to$ Markdown Reports):

```bash
python scripts/run_all.py
```

---

## 3. Step-by-Step Modular Replication

### Step 1: Physical Asset Materialization & Validation
```bash
python scripts/materialize_covers.py
python scripts/materialize_lyrics.py
python scripts/materialize_audio.py
```

### Step 2: Build 12-Class Dataset with Verified `OTHER`
```bash
python scripts/build_12class_dataset.py
```

### Step 3: Generate 5 Distribution-Shift Partitions
```bash
python scripts/create_final12_splits.py
```

### Step 4: Run Data Leakage & Isolation Audit
```bash
python scripts/final12_leakage_audit.py
```

### Step 5: Extract Real Physical Features
```bash
python scripts/extract_features.py
```

### Step 6: Multi-Seed Experiments, Ablation Ladder & Figure Generation
```bash
python scripts/run_master_experiments.py
```

### Step 7: Statistical Significance & Missing Modality Testing
```bash
python scripts/phase8_statistics.py
```

---

## 4. Key Output Artifact Locations

- **Dataset Metadata:** `data/processed/final_12class_metadata.csv`
- **Benchmark Splits:** `data/splits/final12_*.csv`
- **Extracted Features:** `data/features/{lyrics, cover, audio}/`
- **Trained Model Checkpoints:** `outputs/checkpoints/`
- **Numerical Metrics JSON:** `outputs/metrics/final_master_metrics.json`
- **High-Resolution Figures:** `reports/figures/` (12 publication plots)
- **Detailed Markdown Reports:** `reports/*.md`
