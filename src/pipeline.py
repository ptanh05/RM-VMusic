"""Single offline pipeline: validate, prepare, forward-only smoke, train, evaluate.

Training is an explicit CLI subcommand; no preparation/smoke path calls fit_model.
Historical data, caches, reports and checkpoints are never output targets.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import tempfile
import importlib.metadata

# Must be set before any optional CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import (PROJECT_ROOT, GENRE_CLASSES, MODALITIES, SCENARIOS,
                     load_config, resolve_local, file_sha256, json_digest)
from .data.validation import read_metadata, validate_splits
from .data.splits.builder import make_splits, build_all_modular_splits
from .data.feature_extractor import extract_features, inspect_assets
from .data.cache import save_feature_cache, load_feature_cache, write_json
from .data.dataset import FeatureTensorDataset
from .models.uad_fusion import UADFusionModel
from .models.baselines import EarlyConcatModel, LateFusionModel, UnifiedSingleModalityModel
from .training.trainer import set_seed, compute_class_weights, fit_model, predict_model
from .evaluation.metrics import compute_classification_metrics


def environment():
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "Pillow", "PyYAML")
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in packages},
                cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())


def source(config, root):
    path = resolve_local(root, config["dataset"]["metadata"])
    if file_sha256(path) != config["dataset"]["sha256"]:
        raise ValueError("Frozen metadata SHA-256 mismatch; do not mix dataset versions")
    df = read_metadata(path, all_classes=True)
    if len(df) != config["dataset"]["expected_rows"]:
        raise ValueError("Frozen metadata row count mismatch")
    return path, df


def artifact_root(config, root):
    result = resolve_local(root, config["artifacts"])
    if not result.is_relative_to(Path(root).resolve() / "artifacts"):
        raise ValueError("Outputs must remain under artifacts/")
    return result


def data_context(config, scenario):
    implementation = {}
    for path in [Path(__file__).parent / "data/feature_extractor.py",
                 *sorted((Path(__file__).parent / "data/splits").glob("*.py")),
                 Path(__file__).parent / "data/validation.py"]:
        implementation[path.name] = file_sha256(path)
    return dict(pipeline=config["pipeline_version"], metadata_sha256=config["dataset"]["sha256"],
                features=config["features"], split_seed=config["split"]["seed"],
                classes=GENRE_CLASSES, scenario=scenario, implementation=implementation)


def load_partitions(base, scenario):
    return {p: read_metadata(base / "splits" / scenario / f"{p}.csv") for p in ("train", "val", "test")}


def validate_pipeline(config, root=PROJECT_ROOT, require_cache=False):
    _, df = source(config, root)
    generated = make_splits(df, config["split"]["seed"])
    assets = inspect_assets(df, root)
    base = artifact_root(config, root)
    report = dict(status="PASS", metadata_rows=len(df), metadata_sha256=config["dataset"]["sha256"],
                  source_linked_files={m: sum(a["modality"] == m and a["sha256"] is not None for a in assets) for m in MODALITIES},
                  splits={sc: validate_splits(df, parts, sc) for sc, parts in generated.items()},
                  prepared=False, trained=False)
    if require_cache or base.exists():
        manifest_path = base / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Incomplete preparation: manifest absent; choose a fresh artifacts namespace")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["metadata_sha256"] != config["dataset"]["sha256"]:
            raise ValueError("Prepared metadata identity mismatch")
        expected_files = {"metadata.csv"} | {f"splits/{sc}/{part}.csv" for sc in SCENARIOS for part in ("train", "val", "test")} | {"splits/missing_modality/test.csv"} | {f"features/{sc}/manifest.json" for sc in SCENARIOS}
        if set(manifest["files"]) != expected_files:
            raise ValueError("Preparation file manifest is incomplete")
        for name, digest in manifest["files"].items():
            if file_sha256(base / name) != digest:
                raise ValueError(f"Prepared artifact checksum mismatch: {name}")
        if file_sha256(base / "metadata.csv") != config["dataset"]["sha256"]:
            raise ValueError("Prepared metadata copy differs")
        for sc in SCENARIOS:
            parts = load_partitions(base, sc)
            validate_splits(df, parts, sc)
            for name in parts:
                if parts[name].song_id.tolist() != generated[sc][name].song_id.tolist():
                    raise ValueError("Prepared split differs from deterministic generation")
            bundle = load_feature_cache(base / "features" / sc, data_context(config, sc), parts["train"].song_id.tolist())
            if bundle.manifest["assets_sha256"] != json_digest(assets):
                raise ValueError("Physical assets changed since extraction; cache is stale")
            if list(bundle.song_id_map) != df.song_id.tolist():
                raise ValueError("Cache row IDs differ from frozen metadata")
            for array in bundle.features.values():
                array._mmap.close()
        missing = read_metadata(base / "splits/missing_modality/test.csv")
        if not missing.equals(load_partitions(base, "iid")["test"]):
            raise ValueError("Missing-modality benchmark must equal the IID test set")
        report["prepared"] = True
    return report


def prepare(config, root=PROJECT_ROOT):
    path, df = source(config, root)
    base = artifact_root(config, root)
    if base.exists():
        return validate_pipeline(config, root, require_cache=True)
    # Validate every scenario before creating any output.
    make_splits(df, config["split"]["seed"])
    base.mkdir(parents=True, exist_ok=False)
    with (base / "metadata.csv").open("xb") as handle:
        handle.write(path.read_bytes())
    build_all_modular_splits(df, base / "splits", config["split"]["seed"])
    for sc in SCENARIOS:
        parts = load_partitions(base, sc)
        bundle = extract_features(df, parts["train"].song_id.tolist(), root, context=data_context(config, sc))
        save_feature_cache(bundle, base / "features" / sc)
        print(f"Prepared {sc}: coverage={bundle.manifest['coverage']}", flush=True)
        del bundle
    files = [base / "metadata.csv", *sorted((base / "splits").rglob("*.csv")),
             *sorted((base / "features").glob("*/manifest.json"))]
    write_json(base / "manifest.json", dict(pipeline_version=config["pipeline_version"],
               metadata_sha256=config["dataset"]["sha256"], environment=environment(),
               files={p.relative_to(base).as_posix(): file_sha256(p) for p in files}))
    return validate_pipeline(config, root, require_cache=True)


def build_model(name, config):
    common = dict(**{k: config["features"][k] for k in ("lyrics_dim", "cover_dim", "audio_dim")},
                  proj_dim=config["model"]["proj_dim"], num_classes=len(GENRE_CLASSES),
                  dropout=config["model"]["dropout"])
    if name == "uad":
        return UADFusionModel(**common, **{k: config["model"][k] for k in (
            "use_reliability", "use_modality_dropout", "modality_dropout_p")})
    if name == "early_concat":
        return EarlyConcatModel(**common)
    if name == "late_fusion":
        return LateFusionModel(**common)
    if name in ("lyrics_only", "cover_only", "audio_only"):
        mod = name.split("_")[0]
        return UnifiedSingleModalityModel(modality=mod, in_dim=config["features"][f"{mod}_dim"],
               proj_dim=common["proj_dim"], num_classes=common["num_classes"], dropout=common["dropout"])
    raise ValueError(f"Unknown model: {name}")


def smoke(config, root=PROJECT_ROOT, limit=24):
    """Tiny REAL metadata/physical files -> cache -> dataset -> forward -> metrics.

    No optimizer or fitting step. Cache is ephemeral. Report only structural
    success, never untrained scores as experimental results.
    """
    _, df = source(config, root)
    if limit < 4:
        raise ValueError("Smoke limit must be at least 4")
    parts = make_splits(df, config["split"]["seed"])["iid"]
    # Prioritize existing linked lyrics so the test exercises real TF-IDF.
    train = parts["train"].sort_values("lyrics_path", ascending=False).head(max(2, limit // 2))
    test = parts["test"].sort_values("lyrics_path", ascending=False).head(max(2, limit // 2))
    tiny = pd.concat([train, test], ignore_index=True)
    bundle = extract_features(tiny, train.song_id.tolist(), root, context={"purpose": "smoke"})
    with tempfile.TemporaryDirectory(prefix="rmvmusic-smoke-") as temp:
        cache = Path(temp) / "features"
        save_feature_cache(bundle, cache)
        loaded = load_feature_cache(cache, {"purpose": "smoke"}, train.song_id.tolist())
        ds = FeatureTensorDataset.from_bundle(test, loaded)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        set_seed(42, config["training"]["threads"])
        checked = []
        for name in config["training"]["models"]:
            model = build_model(name, config)
            result = predict_model(model, loader, "cpu")
            metrics = compute_classification_metrics(result["labels"], result["probabilities"].argmax(1),
                       result["probabilities"], GENRE_CLASSES, config["evaluation"]["ece_bins"])
            if not np.isfinite(metrics["ece"]):
                raise ValueError("Smoke metric invalid")
            checked.append(name)
        # Close mmap-backed arrays before TemporaryDirectory cleanup on Windows.
        for array in loaded.features.values():
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()
    return dict(status="PASS", purpose="forward-only sanity; not experimental results", samples=len(tiny),
                evaluated_rows=len(ds), models=checked, trained=False, coverage=bundle.manifest["coverage"])


def _run_path(base, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run-id must contain only letters, digits, hyphen, underscore")
    return base / "runs" / run_id


def train(config, root=PROJECT_ROOT, run_id=None):
    validate_pipeline(config, root, require_cache=True)
    base = artifact_root(config, root)
    run = _run_path(base, run_id)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(run / "environment.json", environment())
    train_cfg = config["training"]
    if train_cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent hardware fallback")
    checkpoints = []
    for sc in SCENARIOS:
        parts = load_partitions(base, sc)
        bundle = load_feature_cache(base / "features" / sc, data_context(config, sc), parts["train"].song_id.tolist())
        train_ds = FeatureTensorDataset.from_bundle(parts["train"], bundle)
        val_ds = FeatureTensorDataset.from_bundle(parts["val"], bundle)
        weights = compute_class_weights(train_ds.labels)
        for name in train_cfg["models"]:
            for seed in train_cfg["seeds"]:
                set_seed(seed, train_cfg["threads"])
                model = build_model(name, config)
                generator = torch.Generator().manual_seed(seed)
                train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, generator=generator)
                val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)
                history = fit_model(model, train_loader, val_loader, epochs=train_cfg["epochs"],
                         lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
                         patience=train_cfg["patience"], class_weights=weights, device=train_cfg["device"],
                         is_proposed=name == "uad", lambda_supcon=train_cfg["lambda_supcon"],
                         lambda_inv=train_cfg["lambda_inv"], ece_bins=config["evaluation"]["ece_bins"])
                target = run / f"{sc}__{name}__{seed}.pt"
                torch.save(dict(schema_version=1, model=name, scenario=sc, seed=seed, classes=GENRE_CLASSES,
                           config_sha256=json_digest(config), cache_sha256=file_sha256(base / "features" / sc / "manifest.json"),
                           state_dict={k: v.cpu() for k, v in model.state_dict().items()}), target)
                write_json(target.with_suffix(".history.json"), history)
                checkpoints.append(target.name)
        for array in bundle.features.values():
            array._mmap.close()
    write_json(run / "trained.json", dict(checkpoints={name: file_sha256(run / name) for name in checkpoints}))
    return dict(status="PASS", run=str(run), checkpoints=len(checkpoints))


def evaluate(config, root=PROJECT_ROOT, run_id=None):
    validate_pipeline(config, root, require_cache=True)
    base = artifact_root(config, root)
    run = _run_path(base, run_id)
    saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Evaluation config differs from trained run")
    trained = json.loads((run / "trained.json").read_text(encoding="utf-8"))
    for name, digest in trained["checkpoints"].items():
        if Path(name).name != name or file_sha256(run / name) != digest:
            raise ValueError("Checkpoint checksum mismatch")
    out_dir = run / "evaluation"
    out_dir.mkdir(exist_ok=False)
    records = []
    for filename in trained["checkpoints"]:
        checkpoint = torch.load(run / filename, map_location="cpu", weights_only=True)
        sc = checkpoint["scenario"]
        if checkpoint["classes"] != GENRE_CLASSES or checkpoint["config_sha256"] != json_digest(config):
            raise ValueError("Checkpoint label/config identity mismatch")
        if checkpoint["cache_sha256"] != file_sha256(base / "features" / sc / "manifest.json"):
            raise ValueError("Checkpoint trained on a different cache")
        bundle = load_feature_cache(base / "features" / sc, data_context(config, sc))
        ds = FeatureTensorDataset.from_bundle(load_partitions(base, sc)["test"], bundle)
        loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False)
        model = build_model(checkpoint["model"], config)
        model.load_state_dict(checkpoint["state_dict"])
        rates = config["evaluation"]["missing_rates"] if sc == "iid" else [0.0]
        for rate in rates:
            predictions = predict_model(model, loader, config["training"]["device"], rate, checkpoint["seed"])
            probs, labels = predictions["probabilities"], predictions["labels"]
            metrics = compute_classification_metrics(labels, probs.argmax(1), probs, GENRE_CLASSES, config["evaluation"]["ece_bins"])
            metrics["no_evidence_count"] = int((predictions["masks"].sum(1) == 0).sum())
            stem = f"{Path(filename).stem}__missing_{rate:g}"
            frame = pd.DataFrame(dict(song_id=predictions["song_ids"], label=labels, prediction=probs.argmax(1)))
            for i, genre in enumerate(GENRE_CLASSES):
                frame[f"prob_{genre}"] = probs[:, i]
            for i, mod in enumerate(MODALITIES):
                frame[f"mask_{mod}"] = predictions["masks"][:, i]
                if predictions["weights"].size:
                    frame[f"weight_{mod}"] = predictions["weights"][:, i]
            frame.to_csv(out_dir / f"{stem}.predictions.csv", index=False, mode="x")
            write_json(out_dir / f"{stem}.metrics.json", metrics)
            records.append(dict(scenario=sc, model=checkpoint["model"], seed=checkpoint["seed"], missing_rate=rate,
                           **{k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "ece", "brier_score", "nll")}))
        for array in bundle.features.values():
            array._mmap.close()
    pd.DataFrame(records).to_csv(out_dir / "summary.csv", index=False, mode="x")
    return dict(status="PASS", evaluations=len(records), output=str(out_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "prepare", "smoke", "train", "evaluate"), nargs="?", default="validate")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/official.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("train", "evaluate") and args.run_id is None:
        parser.error("train/evaluate require a unique --run-id")
    config = load_config(args.config)
    if args.stage == "validate":
        result = validate_pipeline(config, require_cache=args.require_cache)
    elif args.stage == "prepare":
        result = prepare(config)
    elif args.stage == "smoke":
        result = smoke(config)
    elif args.stage == "train":
        result = train(config, run_id=args.run_id)
    else:
        result = evaluate(config, run_id=args.run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
