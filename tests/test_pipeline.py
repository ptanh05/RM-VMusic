import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config, validate_config, GENRE_CLASSES
from src.data.splits.builder import make_splits
from src.data.splits.base import verify_zero_artist_leakage
from src.data.validation import validate_splits
from src.evaluation.metrics import compute_classification_metrics, compute_ece
from src import pipeline


def test_split_no_artist_leakage(research_fixture):
    root, config, df = research_fixture
    parts = make_splits(df, 42)["artist_disjoint"]
    assert verify_zero_artist_leakage(parts["train"], parts["val"], parts["test"])
    a = pd.DataFrame({"artist": ["  MIN  "]})
    b = pd.DataFrame({"artist": ["min"]})
    c = pd.DataFrame({"artist": ["Different"]})
    with pytest.raises(ValueError, match="ARTIST LEAKAGE"):
        verify_zero_artist_leakage(a, b, c)


def test_split_reproducibility_and_coverage(research_fixture):
    _, _, df = research_fixture
    first, second = make_splits(df, 42), make_splits(df, 42)
    for sc, parts in first.items():
        validate_splits(df, parts, sc)
        for name, frame in parts.items():
            assert frame.equals(second[sc][name])


def test_split_label_mismatch_fails(research_fixture):
    _, _, df = research_fixture
    parts = make_splits(df, 42)["iid"]
    parts["test"].iloc[0, parts["test"].columns.get_loc("genre")] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_splits(df, parts, "iid")


def test_temporal_requires_verified_year(research_fixture):
    from src.data.splits.temporal import split_temporal
    _, _, df = research_fixture
    with pytest.raises(ValueError, match="verification"):
        split_temporal(df.drop(columns=["year_status"]))


def test_config_loading():
    c = load_config()
    assert c["classes"] == GENRE_CLASSES
    assert c["training"]["device"] == "cpu"
    c["classes"] = list(reversed(GENRE_CLASSES))
    with pytest.raises(ValueError, match="Class order"):
        validate_config(c)


@pytest.mark.parametrize("change", ["unknown", "zero_epochs", "invariance", "bad_seed"])
def test_invalid_config(change):
    c = load_config()
    if change == "unknown":
        c["typo"] = True
    elif change == "zero_epochs":
        c["training"]["epochs"] = 0
    elif change == "invariance":
        c["training"]["lambda_inv"] = .05
    else:
        c["training"]["seeds"] = [-1]
    with pytest.raises(ValueError):
        validate_config(c)


def test_pipeline_smoke(research_fixture, monkeypatch):
    root, config, _ = research_fixture
    def no_train(*a, **kw):
        raise AssertionError("Smoke attempted training")
    monkeypatch.setattr(pipeline, "fit_model", no_train)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", no_train)
    result = pipeline.smoke(config, root, limit=8)
    assert result["status"] == "PASS" and result["trained"] is False
    assert not (root / config["artifacts"]).exists()


def test_prepare_validate_and_stale_cache(research_fixture, monkeypatch):
    root, config, df = research_fixture
    def no_train(*a, **kw):
        raise AssertionError("Prepare attempted training")
    monkeypatch.setattr(pipeline, "fit_model", no_train)
    assert pipeline.prepare(config, root)["prepared"] is True
    assert pipeline.prepare(config, root)["prepared"] is True
    assert pipeline.validate_pipeline(config, root, require_cache=True)["status"] == "PASS"
    (root / "train.txt").write_text("changed physical bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="assets changed"):
        pipeline.validate_pipeline(config, root, require_cache=True)


def test_source_checksum(research_fixture):
    root, config, _ = research_fixture
    with (root / "metadata.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline.validate_pipeline(config, root)


def test_no_output_to_historical_trees(research_fixture):
    root, config, _ = research_fixture
    for target in ["data/features", "reports", "../escape"]:
        config["artifacts"] = target
        with pytest.raises(ValueError):
            pipeline.artifact_root(config, root)


def test_fixed_12_class_metrics():
    labels = np.array([0, 0])
    probabilities = np.zeros((2, 12))
    probabilities[:, 0] = 1
    metrics = compute_classification_metrics(labels, labels, probabilities, GENRE_CLASSES)
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == pytest.approx(1 / 12)
    assert metrics["balanced_accuracy"] == 1
    assert metrics["ece"] == 0 and metrics["brier_score"] == 0
    assert len(metrics["per_class"]) == 12
    assert metrics["per_class"]["OTHER"]["support"] == 0


def test_calibration_known_answer():
    labels = np.array([0, 1])
    p = np.array([[.8, .2], [.8, .2]])
    assert compute_ece(p, labels) == pytest.approx(.3)


@pytest.mark.parametrize("p", [np.array([[np.nan, .5]]), np.array([[.1, .2]]), np.array([[2., -1.]])])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        compute_ece(p, np.array([0]))
