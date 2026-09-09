import shutil
import numpy as np
import pytest
from scipy.io import wavfile

from src.data.feature_extractor import extract_audio, extract_acoustic_descriptor, extract_features


def test_audio_not_fake(media):
    feature, mask, reason = extract_audio(media / "absent.wav")
    assert mask == 0 and not feature.any() and reason == "missing_file"
    with pytest.raises(ValueError, match="ID/genre"):
        extract_acoustic_descriptor("song-id", "ROCK")
    feature, mask, reason = extract_audio(media / "tone.wav")
    assert mask == 1 and reason == "valid"
    assert feature.shape == (128,) and np.isfinite(feature).all()
    assert np.linalg.norm(feature) == pytest.approx(1)


def test_audio_rename_does_not_change_features(media):
    shutil.copyfile(media / "tone.wav", media / "different-id.wav")
    np.testing.assert_array_equal(extract_audio(media / "tone.wav")[0], extract_audio(media / "different-id.wav")[0])


def test_waveform_content_changes_features(media):
    sr = 16000
    signal = (12000 * np.sin(2 * np.pi * 2200 * np.arange(sr) / sr)).astype(np.int16)
    wavfile.write(media / "another.wav", sr, signal)
    assert not np.allclose(extract_audio(media / "tone.wav")[0], extract_audio(media / "another.wav")[0])


@pytest.mark.parametrize("kind", ["silent", "corrupt", "empty", "nan", "constant", "mp3"])
def test_invalid_audio_is_missing(media, kind):
    path = media / f"{kind}.wav"
    if kind == "corrupt":
        path.write_bytes(b"not a waveform")
    elif kind == "empty":
        wavfile.write(path, 16000, np.empty(0, dtype=np.int16))
    elif kind == "nan":
        wavfile.write(path, 16000, np.full(100, np.nan, dtype=np.float32))
    elif kind == "constant":
        wavfile.write(path, 16000, np.full(1000, 100, dtype=np.int16))
    elif kind == "mp3":
        path = media / "song.mp3"
        path.write_bytes(b"unsupported codec")
    feature, mask, reason = extract_audio(path)
    assert mask == 0 and not feature.any() and reason != "valid"


def test_resampling_stereo_and_unsigned_pcm(media):
    sr = 44100
    signal = (10000 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.int16)
    wavfile.write(media / "stereo.wav", sr, np.column_stack([signal, signal]))
    assert extract_audio(media / "stereo.wav")[1] == 1
    u8 = (128 + 100 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.uint8)
    wavfile.write(media / "u8.wav", 16000, u8)
    assert extract_audio(media / "u8.wav")[1] == 1


def test_availability_flag_is_not_evidence(media, small_metadata):
    small_metadata["audio_available"] = 1
    b = extract_features(small_metadata, ["a"], media)
    assert b.masks["audio"].tolist() == [1, 0, 0, 0]


def test_invalid_cover_is_missing(media, small_metadata):
    (media / "cover.jpg").write_bytes(b"invalid jpeg")
    b = extract_features(small_metadata, ["a"], media)
    assert not b.masks["cover"].any()
    assert not b.features["cover"].any()
