"""Strict, offline configuration for the sole supported pipeline."""
from pathlib import Path
import copy
import hashlib
import json
import math

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENRE_CLASSES = [
    "POP_BALLAD", "BOLERO_TRUTINH", "INSTRUMENTAL", "RAP_HIPHOP",
    "FOLK_TRADITIONAL", "DANCE_EDM", "REVOLUTIONARY", "NHAC_TRINH",
    "ROCK", "RB_SOUL", "OTHER", "CHILDREN",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_CLASSES)}
IDX_TO_GENRE = dict(enumerate(GENRE_CLASSES))
MODALITIES = ("lyrics", "cover", "audio")
FEATURE_DIMS = {"lyrics": 5000, "cover": 512, "audio": 128}
SCENARIOS = ("iid", "artist_disjoint", "temporal", "label_shift")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def resolve_local(root, value):
    """Require a local path within the project; never fetch URLs."""
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path must be inside the project: {value}")
    return path


def validate_config(config):
    required = {"schema_version", "pipeline_version", "dataset", "artifacts", "classes",
                "split", "features", "model", "training", "evaluation"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Official config has missing or unknown top-level fields")
    if config["schema_version"] != 1 or config["pipeline_version"] != "official-v1":
        raise ValueError("Unsupported pipeline/schema version")
    if config["classes"] != GENRE_CLASSES:
        raise ValueError("Class order must match the canonical 12-class mapping")
    if config["split"]["scenarios"] != list(SCENARIOS):
        raise ValueError("All four official split scenarios are required")
    if config["features"] != dict(lyrics_dim=5000, cover_dim=512, audio_dim=128,
                                   audio_sample_rate=16000, audio_seconds=10):
        raise ValueError("Feature schema differs from official-v1")
    if not isinstance(config["dataset"]["expected_rows"], int) or config["dataset"]["expected_rows"] < 1:
        raise ValueError("expected_rows must be positive")
    digest = config["dataset"]["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError("Dataset must be frozen by SHA-256")
    train = config["training"]
    if train["lambda_inv"] != 0:
        raise ValueError("Unpaired distribution-invariance loss is not supported")
    if train["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for key in ("threads", "batch_size", "epochs", "patience"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not train["seeds"] or len(set(train["seeds"])) != len(train["seeds"]) or any(type(s) is not int or not 0 <= s < 2**32 for s in train["seeds"]):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    if type(config["split"]["seed"]) is not int or not 0 <= config["split"]["seed"] < 2**32:
        raise ValueError("Invalid split seed")
    for key in ("learning_rate", "weight_decay", "lambda_supcon"):
        value = train[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid training.{key}")
    if train["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    models = {"lyrics_only", "cover_only", "audio_only", "early_concat", "late_fusion", "uad"}
    if not train["models"] or len(set(train["models"])) != len(train["models"]) or not set(train["models"]) <= models:
        raise ValueError("Unknown/duplicate model")
    for key in ("dropout", "modality_dropout_p"):
        if not 0 <= config["model"][key] <= 1:
            raise ValueError(f"Invalid model.{key}")
    if type(config["model"]["proj_dim"]) is not int or config["model"]["proj_dim"] < 1:
        raise ValueError("Invalid projection dimension")
    for key in ("use_reliability", "use_modality_dropout"):
        if type(config["model"][key]) is not bool:
            raise ValueError(f"Invalid model.{key}")
    if type(config["evaluation"]["ece_bins"]) is not int or config["evaluation"]["ece_bins"] < 1:
        raise ValueError("Invalid ece_bins")
    if not config["evaluation"]["missing_rates"] or any(not 0 <= x <= 1 for x in config["evaluation"]["missing_rates"]):
        raise ValueError("Invalid missing rates")
    return config


def load_config(path=None, root=PROJECT_ROOT):
    path = Path(path) if path is not None else Path(root) / "configs/official.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    resolve_local(root, config["dataset"]["metadata"])
    artifacts = resolve_local(root, config["artifacts"])
    # Preparation/training must not overwrite a historical dataset or report tree.
    if not artifacts.is_relative_to(Path(root).resolve() / "artifacts"):
        raise ValueError("Official outputs must be under artifacts/")
    return copy.deepcopy(config)
