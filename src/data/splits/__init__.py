"""
RM-VMusic Modular Splits Package.
Contains specialized splitters for each distribution shift scenario.
"""
from .iid import split_iid
from .artist_disjoint import split_artist_disjoint
from .temporal import split_temporal
from .label_shift import split_label_shift
from .missing_modality import create_missing_modality_benchmark
from .builder import build_all_modular_splits

__all__ = [
    "split_iid",
    "split_artist_disjoint",
    "split_temporal",
    "split_label_shift",
    "create_missing_modality_benchmark",
    "build_all_modular_splits"
]
