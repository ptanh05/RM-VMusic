import copy
import json

import numpy as np
import pytest

from src.config import GENRE_CLASSES, GENRE_TO_IDX, FEATURE_DIMS, MODALITIES
from src.data.dataset import FeatureTensorDataset, RMVMusicDataset
from src.data.cache import save_feature_cache, load_feature_cache
from src.data.feature_extractor import extract_features


def test_label_mapping():
    assert GENRE_CLASSES == ["POP_BALLAD", "BOLERO_TRUTINH", "INSTRUMENTAL", "RAP_HIPHOP",
        "FOLK_TRADITIONAL", "DANCE_EDM", "REVOLUTIONARY", "NHAC_TRINH", "ROCK", "RB_SOUL", "OTHER", "CHILDREN"]
    assert GENRE_TO_IDX["OTHER"] == 10 and GENRE_TO_IDX["CHILDREN"] == 11


def test_dataset_length(bundle, small_metadata):
    ds = FeatureTensorDataset.from_bundle(small_metadata, bundle)
    assert len(ds) == len(ds.labels) == len(ds.song_ids) == 4
    assert len(ds.lyrics_feats) == len(ds.cover_feats) == len(ds.audio_feats) == 4


def test_feature_label_alignment(bundle, small_metadata):
    reordered = small_metadata.iloc[[3, 1, 0]].copy()
    reordered.index = [90, 12, 4]
    ds = FeatureTensorDataset.from_bundle(reordered, bundle)
    for i, sid in enumerate(["d", "b", "a"]):
        assert ds[i]["song_id"] == sid
        assert ds[i]["label"].item() == GENRE_TO_IDX[reordered.iloc[i].genre]
        for mod in MODALITIES:
            np.testing.assert_array_equal(ds[i][f"{mod}_feat"].numpy(), bundle.features[mod][bundle.song_id_map[sid]])


def test_missing_cache_id_fails(bundle, small_metadata):
    small_metadata.loc[1, "song_id"] = "unknown"
    with pytest.raises(ValueError, match="Missing cache"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_explicit_drop_preserves_alignment(bundle, small_metadata):
    small_metadata.loc[1, "song_id"] = "unknown"
    with pytest.warns(UserWarning, match="Explicitly dropped"):
        ds = FeatureTensorDataset.from_bundle(small_metadata, bundle, missing_policy="drop")
    assert ds.song_ids == ["a", "c", "d"]
    assert ds.labels.tolist() == [0, 2, 3]
    assert ds.dropped_song_ids == ["unknown"]


def test_feature_dimensions(bundle):
    for mod, dim in FEATURE_DIMS.items():
        assert bundle.features[mod].shape == (4, dim)
        assert bundle.masks[mod].shape == (4,)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_no_nan_features_and_no_inf_features(bundle, small_metadata, bad):
    bundle.features["lyrics"][0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_invalid_feature_shape(bundle, small_metadata):
    bundle.features["lyrics"] = bundle.features["lyrics"][:, :-1]
    with pytest.raises(ValueError, match="dimensions"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_unknown_genre_rejected(bundle, small_metadata):
    small_metadata.loc[0, "genre"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown genre"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_duplicate_ids_rejected(bundle, small_metadata):
    small_metadata.loc[1, "song_id"] = "a"
    with pytest.raises(ValueError, match="Duplicate"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_non_bijective_feature_map(bundle, small_metadata):
    bundle.song_id_map["b"] = 0
    with pytest.raises(ValueError, match="bijection"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_missing_modality_mask(bundle, small_metadata):
    for mod in MODALITIES:
        assert bundle.masks[mod][2] == 0
        assert not bundle.features[mod][2].any()
    bundle.masks["audio"][1] = 1
    with pytest.raises(ValueError, match="zero feature"):
        FeatureTensorDataset.from_bundle(small_metadata, bundle)


def test_tfidf_train_only(media, small_metadata, bundle):
    assert "heldouttoken" not in bundle.vectorizer_state["vocabulary"]
    assert "exclusiveword" not in bundle.vectorizer_state["vocabulary"]
    (media / "test.txt").write_text("unseentoken testexclusive", encoding="utf-8")
    changed = extract_features(small_metadata, ["a", "d"], media)
    assert bundle.vectorizer_state == changed.vectorizer_state
    np.testing.assert_array_equal(bundle.features["lyrics"][[0, 3]], changed.features["lyrics"][[0, 3]])


def test_no_fallback_when_train_ids_missing(media, small_metadata):
    with pytest.raises(ValueError, match="train IDs"):
        extract_features(small_metadata, [], media)
    with pytest.raises(ValueError, match="train IDs"):
        extract_features(small_metadata, ["not-here"], media)


def test_no_train_vocabulary_is_explicit(media, small_metadata):
    b = extract_features(small_metadata, ["c"], media)
    assert b.manifest["vocabulary_size"] == 0
    assert not b.features["lyrics"].any() and not b.masks["lyrics"].any()


def test_placeholder_removed(bundle, small_metadata):
    with pytest.raises(ValueError, match="cache is required"):
        RMVMusicDataset(small_metadata)
    ds = RMVMusicDataset(small_metadata, feature_bundle=bundle)
    assert ds[0]["audio_feat"].count_nonzero() > 0


def test_cache_roundtrip_and_identity(tmp_path, bundle, small_metadata):
    cache = tmp_path / "cache"
    save_feature_cache(bundle, cache)
    loaded = load_feature_cache(cache, {}, ["a", "d"])
    assert loaded.song_id_map == bundle.song_id_map
    assert len(FeatureTensorDataset.from_bundle(small_metadata, loaded)) == 4
    for x in loaded.features.values():
        x._mmap.close()
    with pytest.raises(ValueError, match="context"):
        load_feature_cache(cache, {"wrong": True})
    with pytest.raises(ValueError, match="train IDs"):
        load_feature_cache(cache, expected_train_ids=["b"])
    with pytest.raises(FileExistsError):
        save_feature_cache(bundle, cache)
    with (cache / "audio_features.npy").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_feature_cache(cache)
