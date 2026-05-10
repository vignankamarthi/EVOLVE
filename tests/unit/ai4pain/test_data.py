"""Tests for ai4pain.data."""
import pytest
import numpy as np
from pathlib import Path
from ai4pain import data


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert callable(data.load_split)


def test_label_map():
    assert data.LABEL_MAP == {"Baseline": 0, "ARM": 1, "HAND": 2}
    assert data.LABEL_NAMES == ["NP", "AP", "HP"]


def test_parse_label_extracts_condition():
    assert data._parse_label("1_Baseline_3") == 0
    assert data._parse_label("12_ARM_5") == 1
    assert data._parse_label("3_HAND_11") == 2
    assert data._parse_label("3_0") is None
    assert data._parse_label("solo") is None


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        data.load_split(tmp_path, split="weird")


def test_load_split_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_split(tmp_path / "nonexistent", split="train")


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present (HIP-A pending)")
def test_load_train_shape_and_balance():
    X, y, subjects = data.load_split(DATA_ROOT, split="train")
    assert len(X) == 41 * 36
    assert y.shape == (41 * 36,)
    assert subjects.shape == (41 * 36,)
    assert all(t.ndim == 2 and t.shape[1] == 4 for t in X[:10])
    assert int((y == 0).sum()) == 41 * 12
    assert int((y == 1).sum()) == 41 * 12
    assert int((y == 2).sum()) == 41 * 12
    assert len(set(subjects.tolist())) == 41


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_load_validation_shape_and_balance():
    X, y, subjects = data.load_split(DATA_ROOT, split="validation")
    assert len(X) == 12 * 36
    assert y.shape == (12 * 36,)
    assert int((y == 0).sum()) == 12 * 12
    assert int((y == 1).sum()) == 12 * 12
    assert int((y == 2).sum()) == 12 * 12
    assert len(set(subjects.tolist())) == 12


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_load_test_is_blinded():
    """ANTIPATTERNS rule 5: test labels are blind. y must be None."""
    X, y, subjects = data.load_split(DATA_ROOT, split="test")
    assert len(X) == 12 * 36
    assert y is None
    assert subjects.shape == (12 * 36,)
    assert len(set(subjects.tolist())) == 12


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_subjects_disjoint_across_splits():
    """ANTIPATTERNS rule 3: no subject in multiple splits."""
    _, _, s_train = data.load_split(DATA_ROOT, split="train")
    _, _, s_val = data.load_split(DATA_ROOT, split="validation")
    _, _, s_test = data.load_split(DATA_ROOT, split="test")
    assert set(s_train.tolist()).isdisjoint(set(s_val.tolist()))
    assert set(s_train.tolist()).isdisjoint(set(s_test.tolist()))
    assert set(s_val.tolist()).isdisjoint(set(s_test.tolist()))


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_signal_channel_order_matches_argument():
    X, _, _ = data.load_split(DATA_ROOT, split="train", signals=("Eda", "Bvp"))
    assert all(t.shape[1] == 2 for t in X[:5])
