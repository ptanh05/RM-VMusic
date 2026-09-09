"""Official split generation and validation; never overwrites existing split files."""
from pathlib import Path

from .iid import split_iid
from .artist_disjoint import split_artist_disjoint
from .temporal import split_temporal
from .label_shift import split_label_shift
from ..validation import validate_metadata, validate_splits, normalize_artist


def make_splits(df_metadata, seed=42):
    validate_metadata(df_metadata, all_classes=True)
    df = df_metadata.copy()
    if "artist" not in df:
        raise ValueError("Artist metadata required")
    df.artist.map(normalize_artist)  # fail on invalid identity before splitting
    splitters = {
        "iid": lambda: split_iid(df, random_state=seed),
        "artist_disjoint": lambda: split_artist_disjoint(df, random_state=seed),
        "temporal": lambda: split_temporal(df),
        "label_shift": lambda: split_label_shift(df, random_state=seed),
    }
    scenarios = {}
    for scenario, build in splitters.items():
        parts = dict(zip(("train", "val", "test"), build()))
        validate_splits(df, parts, scenario)
        scenarios[scenario] = parts
    return scenarios


def build_all_modular_splits(df_metadata, output_root_dir, seed=42):
    scenarios = make_splits(df_metadata, seed)
    root = Path(output_root_dir)
    # Fail before writing ANY file if an expected target already exists.
    targets = [root / scenario / f"{part}.csv" for scenario in scenarios for part in ("train", "val", "test")]
    targets.append(root / "missing_modality/test.csv")
    if any(p.exists() for p in targets):
        raise FileExistsError("Split targets exist; choose a new versioned output directory")
    for scenario, parts in scenarios.items():
        folder = root / scenario
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in parts.items():
            frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8", mode="x")
    (root / "missing_modality").mkdir(parents=True, exist_ok=True)
    scenarios["iid"]["test"].to_csv(root / "missing_modality/test.csv", index=False, encoding="utf-8", mode="x")
    return {sc: {part: len(frame) for part, frame in parts.items()} for sc, parts in scenarios.items()}
