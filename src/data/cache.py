"""Versioned, checksummed, non-pickle physical feature caches."""
import json
from pathlib import Path

import numpy as np

from ..config import MODALITIES, FEATURE_DIMS, file_sha256, json_digest
from .dataset import validate_arrays
from .feature_extractor import FeatureBundle


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def save_feature_cache(bundle, directory):
    directory = Path(directory)
    validate_arrays(bundle.features, bundle.masks, bundle.song_id_map)
    directory.mkdir(parents=True, exist_ok=False)
    for mod in MODALITIES:
        np.save(directory / f"{mod}_features.npy", bundle.features[mod], allow_pickle=False)
        np.save(directory / f"{mod}_masks.npy", bundle.masks[mod], allow_pickle=False)
    ids = [sid for sid, index in sorted(bundle.song_id_map.items(), key=lambda x: x[1])]
    write_json(directory / "song_ids.json", ids)
    write_json(directory / "vectorizer.json", bundle.vectorizer_state)
    manifest = dict(bundle.manifest)
    manifest["files"] = {p.name: file_sha256(p) for p in sorted(directory.iterdir())}
    write_json(directory / "manifest.json", manifest)


def load_feature_cache(directory, expected_context=None, expected_train_ids=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("extractor") != "physical-v1" or manifest.get("dimensions") != FEATURE_DIMS:
        raise ValueError("Unsupported/unverified feature cache; rebuild with the official pipeline")
    if expected_context is not None and manifest["context"] != expected_context:
        raise ValueError("Stale cache: metadata/config/split context mismatch")
    if expected_train_ids is not None and manifest["train_ids"] != sorted(expected_train_ids):
        raise ValueError("Stale cache: TF-IDF train IDs differ")
    required = {f"{mod}_{kind}.npy" for mod in MODALITIES for kind in ("features", "masks")} | {"song_ids.json", "vectorizer.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Cache file manifest is incomplete")
    for name, digest in manifest["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Cache checksum mismatch: {name}")
    if json_digest(manifest["assets"]) != manifest["assets_sha256"]:
        raise ValueError("Asset manifest checksum mismatch")
    ids = json.loads((directory / "song_ids.json").read_text(encoding="utf-8"))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate cache song IDs")
    mapping = {sid: i for i, sid in enumerate(ids)}
    features = {m: np.load(directory / f"{m}_features.npy", mmap_mode="r", allow_pickle=False) for m in MODALITIES}
    masks = {m: np.load(directory / f"{m}_masks.npy", allow_pickle=False) for m in MODALITIES}
    validate_arrays(features, masks, mapping)
    vectorizer = json.loads((directory / "vectorizer.json").read_text(encoding="utf-8"))
    return FeatureBundle(features, masks, mapping, manifest, vectorizer)
