"""
run_all.py
RM-VMusic: Master Reproducibility Pipeline Execution Script.

Runs the complete 7-stage research workflow end-to-end:
1. Physical Asset Materialization (Audio, Covers, Lyrics manifests)
2. Final 12-Class Dataset Construction
3. 5-Split Distribution Shift Generator (IID, Artist Disjoint, Temporal, Label Shift, Missing Modality)
4. Strict Data Leakage & Deduplication Audit
5. Physical Feature Extraction (TF-IDF, Visual Spatial Moments, Zero-Masks)
6. Master Experiments Runner (Multi-Seed Baselines, UAD-Fusion, Ablation, Calibration)
7. Final Quality Gate Verification
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE_STEPS = [
    ("Stage 1: Materialize Physical Audio Manifest", "scripts/materialize_audio.py"),
    ("Stage 2: Materialize Physical Covers Manifest", "scripts/materialize_covers.py"),
    ("Stage 3: Materialize Physical Lyrics Manifest", "scripts/materialize_lyrics.py"),
    ("Stage 4: Build Final 12-Class Dataset with OTHER", "scripts/build_12class_dataset.py"),
    ("Stage 5: Generate All 5 Benchmark Splits", "scripts/create_final12_splits.py"),
    ("Stage 6: Exhaustive Data Leakage Audit", "scripts/final12_leakage_audit.py"),
    ("Stage 7: Generate Modality Matrix", "scripts/generate_modality_matrix.py"),
    ("Stage 8: Extract Real Physical Features", "scripts/extract_features.py"),
    ("Stage 9: Run Master Experiments & Generate Figures", "scripts/run_master_experiments.py")
]

def run_master_pipeline():
    print("================================================================================")
    print("RM-VMusic: Executing Complete End-to-End Research Benchmark Pipeline")
    print("================================================================================\n")
    
    start_total = time.time()
    for name, script_rel in PIPELINE_STEPS:
        script_path = BASE_DIR / script_rel
        print(f"\n>> [RUNNING] {name} ({script_rel})...")
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED: {name} exited with code {res.returncode}")
            sys.exit(res.returncode)
        print(f"   ✓ [COMPLETED] in {elapsed:.2f}s")
        
    total_elapsed = time.time() - start_total
    print("\n================================================================================")
    print(f"SUCCESS: Entire RM-VMusic Research Pipeline Executed in {total_elapsed:.2f}s")
    print("================================================================================")

if __name__ == "__main__":
    run_master_pipeline()
