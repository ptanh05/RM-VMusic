# RM-VMusic Publication Data Package (`reports/paper/`)

Machine-readable CSV tables formatted for direct inclusion in scientific publications.

| File | Description | Source Pipeline |
|---|---|---|
| `paper_dataset_table.csv` | Class distribution and physical asset inventory ($N=5,515$) | `scripts/build_12class_dataset.py` |
| `paper_modality_table.csv` | Per-song physical modality availability matrix | `scripts/generate_modality_matrix.py` |
| `paper_baseline_table.csv` | 7 baseline modality combinations | `scripts/train_physical_baselines.py` |
| `paper_main_results.csv` | 5-seed Mean ± Std across 4 distribution shifts | `scripts/run_master_experiments.py` |
| `paper_shift_results.csv` | Distribution shift performance degradation metrics | `scripts/run_master_experiments.py` |
| `paper_missing_modality.csv` | 11-step granular missing modality stress curve (0% to 100%) | `scripts/phase8_statistics.py` |
| `paper_calibration.csv` | Expected Calibration Error (ECE) across shifts | `scripts/run_master_experiments.py` |
| `paper_ablation.csv` | Model A $	o$ E component ablation ladder | `scripts/run_master_experiments.py` |
| `paper_per_class.csv` | 12-class Precision, Recall, F1, and Support | `scripts/run_master_experiments.py` |
| `paper_statistics.csv` | Paired permutation test p-values and bootstrap CIs | `scripts/phase8_statistics.py` |
