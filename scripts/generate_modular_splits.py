"""
generate_modular_splits.py
CLI Script to generate modular split subdirectories under data/splits/.
"""
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.splits.builder import build_all_modular_splits

def main():
    print("=== Generating Modular Train/Val/Test Splits under data/splits/ ===")
    
    metadata_v3_path = PROJECT_ROOT / "data" / "processed" / "final_12class_metadata_v3.csv"
    if not metadata_v3_path.exists():
        metadata_v3_path = PROJECT_ROOT / "data" / "processed" / "final_12class_metadata_v2.csv"
        
    df = pd.read_csv(metadata_v3_path)
    print(f"Loaded Master Metadata Catalog: N = {len(df):,} samples from {metadata_v3_path.name}")
    
    output_dir = PROJECT_ROOT / "data" / "splits"
    stats = build_all_modular_splits(df, output_dir)
    
    print("\nSuccessfully generated clean modular split folders:")
    print(f"  1. data/splits/iid/              -> Train={stats['iid']['train']:,}, Val={stats['iid']['val']:,}, Test={stats['iid']['test']:,}")
    print(f"  2. data/splits/artist_disjoint/  -> Train={stats['artist_disjoint']['train']:,}, Val={stats['artist_disjoint']['val']:,}, Test={stats['artist_disjoint']['test']:,} [0% LEAKAGE]")
    print(f"  3. data/splits/temporal/         -> Train={stats['temporal']['train']:,}, Val={stats['temporal']['val']:,}, Test={stats['temporal']['test']:,}")
    print(f"  4. data/splits/label_shift/      -> Train={stats['label_shift']['train']:,}, Val={stats['label_shift']['val']:,}, Test={stats['label_shift']['test']:,}")
    print(f"  5. data/splits/missing_modality/ -> Test={stats['missing_modality']['test']:,}")
    print("\nAll modular splits ready for production and benchmarking!")

if __name__ == "__main__":
    main()
