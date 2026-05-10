"""Subject-disjoint splits and K-subject subset sampler.

ANTIPATTERNS rule 3: per-subject splits only. No subject in multiple splits.
ANTIPATTERNS rule 19: K-subject subset becomes the loop fitness set only after
                     HIP-B selects K from the subset-transfer experiment.

Both functions are deterministic given a seed.
"""
from typing import Sequence, TypeVar
import numpy as np


T = TypeVar("T")


def subject_disjoint_split(all_subjects: Sequence[T], n_val: int,
                           seed: int = 0) -> tuple[list[T], list[T]]:
    """Random subject-disjoint train and val partition.

    Args:
        all_subjects: ordered sequence of subject IDs (any hashable type).
        n_val: number of subjects to assign to val.
        seed: RNG seed for determinism.

    Returns:
        (train_subjects, val_subjects). Lists. Train preserves the original order
        of non-selected subjects. Val preserves the order of the permutation pick.
    """
    n = len(all_subjects)
    if n_val < 0 or n_val > n:
        raise ValueError(f"n_val={n_val} invalid for {n} subjects")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = set(perm[:n_val].tolist())
    train = [all_subjects[i] for i in range(n) if i not in val_idx]
    val = [all_subjects[i] for i in perm[:n_val].tolist()]
    return train, val


def k_subject_subset(train_subjects: Sequence[T], k: int,
                     seed: int = 0) -> list[T]:
    """Random K-subject subset of training subjects.

    Used by framework.eval as the inner-loop fitness evaluation set after HIP-B.

    Args:
        train_subjects: ordered sequence of training subject IDs.
        k: number of subjects to sample (without replacement).
        seed: RNG seed for determinism.

    Returns:
        List of K subject IDs.
    """
    n = len(train_subjects)
    if k < 0 or k > n:
        raise ValueError(f"k={k} invalid for {n} train subjects")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [train_subjects[i] for i in perm[:k].tolist()]
