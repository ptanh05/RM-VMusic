"""Offline, deterministic fixtures. Synthetic tones are tests, never research data."""
import copy
import socket

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.io import wavfile

from src.config import GENRE_CLASSES, load_config, file_sha256


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def media(tmp_path):
    (tmp_path / "train.txt").write_text("tinh yeu que huong am nhac", encoding="utf-8")
    (tmp_path / "test.txt").write_text("heldouttoken exclusiveword tinh yeu", encoding="utf-8")
    Image.new("RGB", (20, 20), color=(40, 100, 220)).save(tmp_path / "cover.jpg")
    sr = 16000
    signal = (12000 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.int16)
    wavfile.write(tmp_path / "tone.wav", sr, signal)
    wavfile.write(tmp_path / "silent.wav", sr, np.zeros(sr, dtype=np.int16))
    return tmp_path


@pytest.fixture
def small_metadata(media):
    return pd.DataFrame([
        dict(song_id="a", title="A", artist="Artist A", genre=GENRE_CLASSES[0], lyrics_path="train.txt", cover_path="cover.jpg", audio_path="tone.wav"),
        dict(song_id="b", title="B", artist="Artist B", genre=GENRE_CLASSES[1], lyrics_path="test.txt", cover_path="", audio_path="absent.wav"),
        dict(song_id="c", title="C", artist="Artist C", genre=GENRE_CLASSES[2], lyrics_path="", cover_path="", audio_path=""),
        dict(song_id="d", title="D", artist="Artist D", genre=GENRE_CLASSES[3], lyrics_path="train.txt", cover_path="cover.jpg", audio_path="silent.wav"),
    ])


@pytest.fixture
def bundle(media, small_metadata):
    from src.data.feature_extractor import extract_features
    return extract_features(small_metadata, ["a", "d"], media)


@pytest.fixture
def research_fixture(media):
    """Enough groups/classes to exercise all official splitters, without training."""
    rows = []
    for c, genre in enumerate(GENRE_CLASSES):
        for artist in range(16):
            for track in range(2):
                rows.append(dict(song_id=f"{c}-{artist}-{track}", title=f"Song {c}-{artist}-{track}",
                    artist=f"Artist {c}-{artist}", genre=genre, lyrics_path="train.txt",
                    cover_path="cover.jpg", audio_path="tone.wav" if track else "",
                    release_year=[2010, 2019, 2022][artist % 3], year_status="verified"))
    df = pd.DataFrame(rows)
    path = media / "metadata.csv"
    df.to_csv(path, index=False)
    config = copy.deepcopy(load_config())
    config["dataset"] = dict(metadata="metadata.csv", sha256=file_sha256(path), expected_rows=len(df))
    config["model"]["proj_dim"] = 8
    config["training"]["models"] = ["early_concat", "late_fusion", "uad"]
    return media, config, df
