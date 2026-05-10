"""Tests for ai4pain.splits."""
import pytest
from ai4pain import splits


def test_module_imports():
    assert callable(splits.subject_disjoint_split)
    assert callable(splits.k_subject_subset)


def test_subject_disjoint_split_no_overlap():
    train, val = splits.subject_disjoint_split(
        all_subjects=[f"S{i:02d}" for i in range(41)],
        n_val=5, seed=0,
    )
    assert set(train).isdisjoint(set(val))
    assert len(train) == 36
    assert len(val) == 5


def test_subject_disjoint_split_deterministic():
    a = splits.subject_disjoint_split(list(range(41)), n_val=5, seed=42)
    b = splits.subject_disjoint_split(list(range(41)), n_val=5, seed=42)
    assert a == b


def test_subject_disjoint_split_zero_val():
    train, val = splits.subject_disjoint_split([1, 2, 3], n_val=0, seed=0)
    assert val == []
    assert sorted(train) == [1, 2, 3]


def test_subject_disjoint_split_invalid_n_val():
    with pytest.raises(ValueError):
        splits.subject_disjoint_split([1, 2, 3], n_val=10, seed=0)
    with pytest.raises(ValueError):
        splits.subject_disjoint_split([1, 2, 3], n_val=-1, seed=0)


def test_k_subject_subset_size_and_uniqueness():
    chosen = splits.k_subject_subset(
        train_subjects=[f"S{i:02d}" for i in range(41)],
        k=10, seed=0,
    )
    assert len(chosen) == 10
    assert len(set(chosen)) == 10


def test_k_subject_subset_deterministic():
    a = splits.k_subject_subset(list(range(41)), k=10, seed=7)
    b = splits.k_subject_subset(list(range(41)), k=10, seed=7)
    assert a == b


def test_k_subject_subset_invalid_k():
    with pytest.raises(ValueError):
        splits.k_subject_subset([1, 2, 3], k=10, seed=0)
    with pytest.raises(ValueError):
        splits.k_subject_subset([1, 2, 3], k=-1, seed=0)


def test_k_subject_subset_zero_k_returns_empty():
    assert splits.k_subject_subset([1, 2, 3], k=0, seed=0) == []
