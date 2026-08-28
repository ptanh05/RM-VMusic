"""
Master Split Builder Module.
Constructs and exports modular split folders under data/splits/<scenario>/.
"""
import pandas as pd
from pathlib import Path
from .iid import split_iid
from .artist_disjoint import split_artist_disjoint
from .temporal import split_temporal
from .label_shift import split_label_shift
from .missing_modality import create_missing_modality_benchmark

def build_all_modular_splits(df_metadata, output_root_dir):
    """
    Builds and writes all 5 benchmark scenarios into structured subdirectories:
    - data/splits/iid/
    - data/splits/artist_disjoint/
    - data/splits/temporal/
    - data/splits/label_shift/
    - data/splits/missing_modality/
    """
    root_path = Path(output_root_dir)
    
    # 1. IID Split
    dir_iid = root_path / "iid"
    dir_iid.mkdir(parents=True, exist_ok=True)
    tr_iid, va_iid, te_iid = split_iid(df_metadata)
    tr_iid.to_csv(dir_iid / "train.csv", index=False)
    va_iid.to_csv(dir_iid / "val.csv", index=False)
    te_iid.to_csv(dir_iid / "test.csv", index=False)
    
    # 2. Artist Disjoint Split
    dir_ad = root_path / "artist_disjoint"
    dir_ad.mkdir(parents=True, exist_ok=True)
    tr_ad, va_ad, te_ad = split_artist_disjoint(df_metadata)
    tr_ad.to_csv(dir_ad / "train.csv", index=False)
    va_ad.to_csv(dir_ad / "val.csv", index=False)
    te_ad.to_csv(dir_ad / "test.csv", index=False)
    
    # 3. Temporal Split
    dir_temp = root_path / "temporal"
    dir_temp.mkdir(parents=True, exist_ok=True)
    tr_temp, va_temp, te_temp = split_temporal(df_metadata)
    tr_temp.to_csv(dir_temp / "train.csv", index=False)
    va_temp.to_csv(dir_temp / "val.csv", index=False)
    te_temp.to_csv(dir_temp / "test.csv", index=False)
    
    # 4. Label Shift Split
    dir_ls = root_path / "label_shift"
    dir_ls.mkdir(parents=True, exist_ok=True)
    tr_ls, va_ls, te_ls = split_label_shift(df_metadata)
    tr_ls.to_csv(dir_ls / "train.csv", index=False)
    va_ls.to_csv(dir_ls / "val.csv", index=False)
    te_ls.to_csv(dir_ls / "test.csv", index=False)
    
    # 5. Missing Modality Benchmark
    dir_mm = root_path / "missing_modality"
    dir_mm.mkdir(parents=True, exist_ok=True)
    te_mm = create_missing_modality_benchmark(te_iid)
    te_mm.to_csv(dir_mm / "test.csv", index=False)
    
    return {
        "iid": {"train": len(tr_iid), "val": len(va_iid), "test": len(te_iid)},
        "artist_disjoint": {"train": len(tr_ad), "val": len(va_ad), "test": len(te_ad)},
        "temporal": {"train": len(tr_temp), "val": len(va_temp), "test": len(te_temp)},
        "label_shift": {"train": len(tr_ls), "val": len(va_ls), "test": len(te_ls)},
        "missing_modality": {"test": len(te_mm)}
    }
