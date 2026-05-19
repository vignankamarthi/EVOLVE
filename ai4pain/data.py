"""AI4Pain 2026 data loader.

Reads `data/raw/{train,validation,test}/{Bvp,Eda,Resp,SpO2}/<subj>.csv`.

CSV layout (per file):
  - columns are trials, rows are time samples
  - column header encodes labels for train and validation:
      <subj>_Baseline_<n>  -> label 0 (No Pain)
      <subj>_ARM_<n>       -> label 1 (Arm Pain)
      <subj>_HAND_<n>      -> label 2 (Hand Pain)
  - test column header is blinded: <subj>_<idx>
  - 36 trials per subject (12 of each class for train and validation)
  - trial length varies per subject (e.g. 1022 or 1118 samples)
  - all 4 signals share the same shape per subject

ANTIPATTERNS rule 5: test labels are blind. load_split returns y=None for test.
"""
from __future__ import annotations  # PEP 604 (int | None) safe on Python 3.9

from pathlib import Path
import csv
import numpy as np


# Trial label encoding: 0=No Pain, 1=Arm Pain, 2=Hand Pain
LABEL_MAP: dict[str, int] = {"Baseline": 0, "ARM": 1, "HAND": 2}
LABEL_NAMES: list[str] = ["NP", "AP", "HP"]


def _parse_label(column_name: str) -> int | None:
    """Extract label from a labeled column name. Returns None if not labeled."""
    parts = column_name.split("_")
    if len(parts) < 2:
        return None
    return LABEL_MAP.get(parts[1])


def _load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Load one CSV. Returns (column_names, samples) where samples is (T, n_trials) float32."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, np.asarray(rows, dtype=np.float32)


def load_split(data_root: Path, split: str,
               signals: tuple[str, ...] = ("Bvp", "Eda", "Resp", "SpO2")
               ) -> tuple[list[np.ndarray], np.ndarray | None, np.ndarray]:
    """Load one split.

    Args:
        data_root: project's `data/raw/` root.
        split: 'train', 'validation', or 'test'.
        signals: tuple of signal subdirectory names; channel order in returned arrays
                 follows this tuple.

    Returns:
        X: list of length N, each element is a (T_i, C) float32 ndarray for one trial,
           where C = len(signals). T_i may vary per subject.
        y: int64 ndarray of shape (N,) with labels in {0, 1, 2}, OR None for split='test'.
        subjects: int64 ndarray of shape (N,) with subject IDs aligned to X and y.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split!r}")

    data_root = Path(data_root)
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")

    is_test = split == "test"

    first_sig_dir = split_dir / signals[0]
    if not first_sig_dir.is_dir():
        raise FileNotFoundError(f"signal directory not found: {first_sig_dir}")
    subject_ids = sorted(int(p.stem) for p in first_sig_dir.glob("*.csv"))
    if not subject_ids:
        raise FileNotFoundError(f"no CSV files in {first_sig_dir}")

    X: list[np.ndarray] = []
    y_list: list[int] = []
    subj_list: list[int] = []

    for subj in subject_ids:
        per_signal: dict[str, np.ndarray] = {}
        ref_header: list[str] | None = None
        ref_shape: tuple[int, int] | None = None

        for sig in signals:
            path = split_dir / sig / f"{subj}.csv"
            header, arr = _load_csv(path)
            if ref_header is None:
                ref_header = header
                ref_shape = arr.shape
            else:
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"signal {sig} for subject {subj} has shape {arr.shape}, "
                        f"expected {ref_shape}")
            per_signal[sig] = arr

        # (T, n_trials, C)
        stacked = np.stack([per_signal[s] for s in signals], axis=-1)
        n_trials = stacked.shape[1]

        for trial_idx in range(n_trials):
            X.append(stacked[:, trial_idx, :])
            subj_list.append(subj)
            if not is_test:
                label = _parse_label(ref_header[trial_idx])
                if label is None:
                    raise ValueError(
                        f"unparseable label in column {ref_header[trial_idx]!r} "
                        f"for subject {subj}")
                y_list.append(label)

    y = None if is_test else np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subj_list, dtype=np.int64)
    return X, y, subjects
