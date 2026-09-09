"""Physical-only extraction. No ID, URL, genre or random vector is an audio input."""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import FEATURE_DIMS, MODALITIES, file_sha256, json_digest, resolve_local
from .validation import validate_metadata


@dataclass
class FeatureBundle:
    features: dict
    masks: dict
    song_id_map: dict
    manifest: dict
    vectorizer_state: dict


def extract_visual_descriptor(img_path):
    """512 dims: 432 spatial RGB bins + RGB mean/std + 74 gradient bins."""
    with Image.open(img_path) as img:
        arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    values = []
    for row in np.array_split(arr, 3, axis=0):
        for patch in np.array_split(row, 3, axis=1):
            for ch in range(3):
                hist = np.histogram(patch[:, :, ch], bins=16, range=(0, 1))[0]
                values.extend(hist / hist.sum())
    gray = arr @ np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
    dy, dx = np.gradient(gray)
    grad = np.histogram(np.hypot(dx, dy), bins=74, range=(0, 1))[0]
    grad = grad / max(grad.sum(), 1)
    feat = np.concatenate([values, arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), grad]).astype(np.float32)
    return feat / np.linalg.norm(feat)


def extract_acoustic_descriptor(audio_path, genre_name=None):
    """Decode WAV, mono 16 kHz, first <=10 s; 128 log spectral bands.

    The obsolete (song_id, genre) signature fails explicitly. Silent/empty/
    non-finite WAV fails. MP3 is explicitly unsupported, not silently decoded.
    """
    if genre_name is not None:
        raise ValueError("ID/genre audio features were removed; provide a physical WAV path")
    path = Path(audio_path)
    if path.suffix.lower() != ".wav":
        raise ValueError("unsupported_audio_format: only WAV is supported")
    sr, signal = wavfile.read(path)
    if sr <= 0 or signal.ndim not in (1, 2) or signal.shape[0] < 2:
        raise ValueError("invalid_waveform")
    signal = signal[:int(sr * 10)]
    dtype = signal.dtype
    x = signal.astype(np.float64)
    if dtype == np.uint8:
        x = (x - 128) / 128
    elif np.issubdtype(dtype, np.integer):
        x /= max(abs(np.iinfo(dtype).min), np.iinfo(dtype).max)
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError("unsupported_waveform_dtype")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_waveform")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) <= 1e-8:
        raise ValueError("silent_waveform")
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        x = resample_poly(x, 16000 // divisor, sr // divisor)
    x -= x.mean()
    if np.max(np.abs(x)) <= 1e-8:
        raise ValueError("constant_waveform")
    x = x / np.max(np.abs(x))
    if len(x) < 512:
        x = np.pad(x, (0, 512 - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
    power = np.abs(np.fft.rfft(frames * np.hanning(512), axis=1)) ** 2
    spectrum = power.mean(axis=0)[1:]
    feat = np.log1p(spectrum.reshape(128, 2).mean(axis=1)).astype(np.float32)
    norm = np.linalg.norm(feat)
    if not np.isfinite(feat).all() or norm <= 0:
        raise ValueError("invalid_audio_feature")
    return feat / norm


def extract_audio(audio_path):
    """Return (feature, mask, reason); mask=1 requires successful physical decode."""
    zero = np.zeros(128, dtype=np.float32)
    if audio_path is None or not Path(audio_path).is_file():
        return zero, 0.0, "missing_file"
    if Path(audio_path).suffix.lower() != ".wav":
        return zero, 0.0, "unsupported_audio_format"
    try:
        return extract_acoustic_descriptor(audio_path), 1.0, "valid"
    except (ValueError, OSError, EOFError) as exc:
        return zero, 0.0, f"invalid_audio:{type(exc).__name__}"


def inspect_assets(metadata, root):
    """Hash linked bytes, not availability flags; missing paths have no digest."""
    assets = []
    for row in metadata.to_dict("records"):
        for mod in MODALITIES:
            raw = row.get(f"{mod}_path", "")
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                raw = ""
            raw = str(raw).strip()
            path = resolve_local(root, raw) if raw else None
            present = path is not None and path.is_file()
            assets.append(dict(song_id=row["song_id"], modality=mod, path=raw,
                               sha256=file_sha256(path) if present else None))
    return assets


def extract_features(metadata, train_ids, root, *, context=None):
    validate_metadata(metadata)
    train_ids = list(train_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids) or not set(train_ids) <= set(metadata.song_id):
        raise ValueError("Explicit unique train IDs from this metadata are required")
    assets = inspect_assets(metadata, root)
    asset_lookup = {(a["song_id"], a["modality"]): a for a in assets}
    n = len(metadata)
    feats = {mod: np.zeros((n, dim), dtype=np.float32) for mod, dim in FEATURE_DIMS.items()}
    masks = {mod: np.zeros(n, dtype=np.float32) for mod in MODALITIES}
    statuses, texts = [], []
    for i, sid in enumerate(metadata.song_id):
        for mod in MODALITIES:
            asset = asset_lookup[sid, mod]
            path = resolve_local(root, asset["path"]) if asset["path"] else None
            reason = "missing_file"
            if mod == "lyrics":
                text = ""
                if asset["sha256"]:
                    try:
                        text = path.read_text(encoding="utf-8").strip()
                        masks[mod][i] = float(bool(text))
                        reason = "valid" if text else "empty_lyrics"
                    except (OSError, UnicodeError):
                        reason = "invalid_lyrics"
                texts.append(text)
            elif mod == "cover" and asset["sha256"]:
                try:
                    feat = extract_visual_descriptor(path)
                    if feat.shape != (512,) or not np.isfinite(feat).all() or np.linalg.norm(feat) == 0:
                        raise ValueError("invalid_cover_feature")
                    feats[mod][i], masks[mod][i], reason = feat, 1.0, "valid"
                except (OSError, ValueError, UnidentifiedImageError):
                    reason = "invalid_cover"
            elif mod == "audio":
                feats[mod][i], masks[mod][i], reason = extract_audio(path)
            statuses.append(dict(song_id=sid, modality=mod, status=reason))
    train_set = set(train_ids)
    train_texts = [text for sid, text in zip(metadata.song_id, texts) if sid in train_set and text]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
                               dtype=np.float32)
    vocabulary, idf = {}, []
    if train_texts:
        analyzer = vectorizer.build_analyzer()
        if any(analyzer(t) for t in train_texts):
            vectorizer.fit(train_texts)
            transformed = vectorizer.transform(texts).toarray()
            feats["lyrics"][:, :transformed.shape[1]] = transformed
            vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
            idf = vectorizer.idf_.tolist()
    if not vocabulary:
        masks["lyrics"][:] = 0
        for status in statuses:
            if status["modality"] == "lyrics" and status["status"] == "valid":
                status["status"] = "no_train_vocabulary"
    manifest = dict(schema_version=1, extractor="physical-v1", dimensions=FEATURE_DIMS,
                    context=context or {}, train_ids=sorted(train_ids), assets=assets,
                    assets_sha256=json_digest(assets), statuses=statuses,
                    coverage={mod: int(masks[mod].sum()) for mod in MODALITIES},
                    vocabulary_size=len(vocabulary))
    return FeatureBundle(feats, masks, {sid: i for i, sid in enumerate(metadata.song_id)},
                         manifest, dict(vocabulary=vocabulary, idf=idf))


def extract_all_v4_features():
    raise RuntimeError("V4 extraction is retired. Use python scripts/run_all.py prepare")


if __name__ == "__main__":
    extract_all_v4_features()
