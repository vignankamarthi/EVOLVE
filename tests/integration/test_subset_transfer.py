"""Integration: subset-transfer experiment exercises ai4pain.splits + framework.eval.

ANTIPATTERNS rule 19: loop cannot run on K-subject subset until this experiment
selects K via HIP-B. This integration test asserts the workflow produces a
correlation table and the subject-disjointness invariant holds.
"""
import pytest
from pathlib import Path
from framework import eval as feval
from ai4pain import splits


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red, awaiting HIP-A")
def test_subset_transfer_produces_correlation_table(tmp_path: Path, sample_program_spec):
    table = feval.subset_transfer_experiment(
        baseline_spec=sample_program_spec,
        k_grid=[5, 10, 15, 20],
        n_seeds=3,
        out_dir=tmp_path,
    )
    assert "correlations" in table
    assert "chosen_k" in table
    assert table["chosen_k"] in (5, 10, 15, 20)


def test_subset_subjects_disjoint_from_val():
    """ai4pain.splits is implemented, this is now a green-state invariant test."""
    all_subjects = [f"S{i:02d}" for i in range(41)]
    train, val = splits.subject_disjoint_split(all_subjects, n_val=5, seed=0)
    chosen = splits.k_subject_subset(train, k=10, seed=0)
    assert set(chosen).isdisjoint(set(val))
    assert set(chosen).issubset(set(train))
