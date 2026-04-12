"""AI4Pain 2026 data loading and preprocessing.

Adapted from `Blood-Pressure-Inference-with-BVP/src/data_loader.py`. Key
differences from the BP inference version:

  - **Multi-signal join**: the Rust extractor produces one CSV per
    (split, signal) pair (`results_{split}_{signal}.csv`). For an ablation
    config like ``bvp_eda``, we join the bvp and eda CSVs by
    ``(file_name, segment_id)`` to produce a wide feature matrix with 80
    columns (40 features per signal, two signals).

  - **3-class labels**: BP inference predicted continuous SBP and DBP from a
    single ABP-derived label. AI4Pain 2026 has 3-class trial labels
    (NP / HP / AP) joined from a labels metadata file.

  - **Subject-level splits**: the challenge provides a fixed 41 / 12 / 12
    subject split. We never re-split, but we still preserve ``subject_id`` so
    Optuna's stratified-by-subject CV inside the training split is honored
    (no subject in two folds).

  - **Synthetic data**: ``--dry-run`` mode constructs a small synthetic feature
    matrix that exercises the entire pipeline without requiring real data.
    This is the basis for Phase 2's verification criterion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import get_logger


logger = get_logger("ai4pain_2026")


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

LABEL_MAP = {"NP": 0, "HP": 1, "AP": 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["No Pain", "Hand Pain", "Arm Pain"]
N_CLASSES = 3

SPLIT_SIZES = {"train": 41, "validation": 12, "test": 12}
SPLIT_NAMES = ("train", "validation", "test")

# Metadata columns produced by the Rust extractor (excluded from the feature matrix)
METADATA_COLS = {"file_name", "segment_id", "signal_length", "nan_percentage"}

# Feature column counts (per signal)
N_FEATURES_PER_SIGNAL = 40


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------


@dataclass
class AI4PainSplit:
    """One split of the AI4Pain 2026 dataset after feature extraction + joining.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Integer labels of shape ``(n_samples,)``, values in {0, 1, 2}.
    subject_ids : np.ndarray
        Subject identifiers, shape ``(n_samples,)``. Used for subject-level CV
        inside the training split.
    trial_ids : np.ndarray
        Trial identifiers, shape ``(n_samples,)``. Useful for joining
        predictions back to source rows.
    feature_names : list[str]
        Length-``n_features`` list of column names. Order matches X columns.
    """

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    feature_names: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0

    @property
    def class_distribution(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        return {INVERSE_LABEL_MAP[int(u)]: int(c) for u, c in zip(unique, counts)}


# ----------------------------------------------------------------------------
# Real data loading
# ----------------------------------------------------------------------------


def load_signal_features_csv(path: Path) -> pd.DataFrame:
    """Load a single Rust-extracted feature CSV for one (split, signal) pair."""
    logger.info(f"Loading features from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the feature columns (excluding metadata)."""
    return [c for c in df.columns if c not in METADATA_COLS]


def join_signals(
    features_dir: Path,
    split: str,
    signals: list,
) -> pd.DataFrame:
    """Join the per-signal feature CSVs for one ablation config and split.

    The join key is ``(file_name, segment_id)``. Feature columns are renamed to
    ``{signal}__{feature}`` so the wide matrix has 40 * len(signals) feature
    columns plus the metadata key columns.

    Parameters
    ----------
    features_dir : Path
        Directory containing ``results_{split}_{signal}.csv`` files (the
        Rust extractor output).
    split : str
        ``'train'``, ``'validation'``, or ``'test'``.
    signals : list[str]
        Signals to join, e.g. ``['bvp', 'eda']``. Order matters for column
        order in the final matrix.

    Returns
    -------
    pd.DataFrame
        Joined feature dataframe.

    Raises
    ------
    FileNotFoundError
        If any expected per-signal CSV is missing.
    ValueError
        If the join leaves zero rows (likely indicates inconsistent
        ``(file_name, segment_id)`` keys across signals).
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    if not signals:
        raise ValueError("signals list cannot be empty")

    joined = None
    for signal in signals:
        csv_path = features_dir / f"results_{split}_{signal}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing feature CSV for signal '{signal}' split '{split}': {csv_path}"
            )

        df = load_signal_features_csv(csv_path)
        feature_cols = get_feature_columns(df)
        renamed = {c: f"{signal}__{c}" for c in feature_cols}
        df = df.rename(columns=renamed)

        if joined is None:
            joined = df
        else:
            # Inner join on the metadata keys; drop the duplicated metadata
            # columns that come from the right side (signal_length, nan_pct
            # vary per signal so they are signal-specific too)
            right_drop = ["signal_length", "nan_percentage"]
            df_right = df.drop(columns=right_drop, errors="ignore")
            joined = joined.merge(
                df_right,
                on=["file_name", "segment_id"],
                how="inner",
                suffixes=("", f"__{signal}"),
            )

    if joined is None or len(joined) == 0:
        raise ValueError(
            f"join_signals produced empty dataframe for split={split} signals={signals}. "
            "Check that all signal CSVs share the same (file_name, segment_id) keys."
        )

    logger.info(
        f"Joined {len(signals)} signals for split={split}: {joined.shape[0]} rows, {joined.shape[1]} columns"
    )
    return joined


def attach_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Attach trial labels and subject_ids to a joined feature dataframe.

    The labels dataframe is expected to have columns
    ``(file_name, subject_id, label)`` where label is one of
    {``'NP'``, ``'HP'``, ``'AP'``}. Format is confirmed at RT-A.
    """
    required = {"file_name", "subject_id", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_df missing required columns: {missing}")

    merged = features_df.merge(labels_df, on="file_name", how="inner")
    merged["label_int"] = merged["label"].map(LABEL_MAP)
    if merged["label_int"].isna().any():
        bad = merged.loc[merged["label_int"].isna(), "label"].unique().tolist()
        raise ValueError(f"Unknown labels encountered: {bad}. Expected NP, HP, AP.")
    return merged


def split_to_arrays(
    df: pd.DataFrame,
    feature_cols: list,
) -> AI4PainSplit:
    """Convert a joined+labeled dataframe into an ``AI4PainSplit``."""
    X = df[feature_cols].to_numpy(dtype=np.float64, copy=True)
    y = df["label_int"].to_numpy(dtype=np.int64)
    subject_ids = df["subject_id"].to_numpy()
    trial_ids = df["segment_id"].to_numpy()
    return AI4PainSplit(
        X=X,
        y=y,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        feature_names=list(feature_cols),
    )


def load_split(
    features_dir: Path,
    labels_path: Path,
    split: str,
    signals: list,
) -> AI4PainSplit:
    """Load one split end to end: join signals, attach labels, build the split.

    Parameters
    ----------
    features_dir : Path
        Directory holding ``results_{split}_{signal}.csv`` files.
    labels_path : Path
        Path to the label metadata file (CSV with file_name, subject_id, label).
    split : str
        ``'train'`` / ``'validation'`` / ``'test'``.
    signals : list[str]
        Signals to include, in order. The number of feature columns is
        ``40 * len(signals)``.

    Returns
    -------
    AI4PainSplit
    """
    joined = join_signals(features_dir, split, signals)
    labels_df = pd.read_csv(labels_path)
    labeled = attach_labels(joined, labels_df)
    feature_cols = [c for c in labeled.columns if c.startswith(tuple(f"{s}__" for s in signals))]
    return split_to_arrays(labeled, feature_cols)


# ----------------------------------------------------------------------------
# Synthetic data (dry-run mode)
# ----------------------------------------------------------------------------


def make_synthetic_split(
    n_subjects: int,
    n_trials_per_subject: int,
    n_features: int,
    seed: int = 42,
) -> AI4PainSplit:
    """Build a small synthetic split that respects the subject-level split rule.

    The labels are sampled per-trial uniformly across the 3 classes; the
    features are gaussian noise with a tiny class-conditional mean shift so
    that classifiers can plausibly distinguish them. This is the data that
    powers ``--dry-run`` mode and the smoke tests.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_subjects * n_trials_per_subject

    y = rng.integers(0, N_CLASSES, size=n_samples)
    subject_ids = np.repeat([f"S{i:03d}" for i in range(n_subjects)], n_trials_per_subject)
    trial_ids = np.array([f"T{i:04d}" for i in range(n_samples)])

    X = rng.standard_normal(size=(n_samples, n_features))
    # Mild class-conditional shift so HPO has signal to optimize against
    for cls in range(N_CLASSES):
        mask = y == cls
        X[mask] += 0.3 * (cls - 1)

    feature_names = [f"synth_feat_{i:03d}" for i in range(n_features)]
    return AI4PainSplit(
        X=X,
        y=y,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        feature_names=feature_names,
    )


# ----------------------------------------------------------------------------
# StandardScaler fitting
# ----------------------------------------------------------------------------


def fit_scaler_on_train(
    train_split: AI4PainSplit,
    checkpoint_dir: Path,
    force_refit: bool = False,
) -> StandardScaler:
    """Fit (or load cached) StandardScaler on the training feature matrix only.

    The scaler is persisted to ``checkpoint_dir / scaler.pkl``. Subsequent runs
    reuse it unless ``force_refit=True``.
    """
    scaler_path = checkpoint_dir / "scaler.pkl"
    if scaler_path.exists() and not force_refit:
        logger.info(f"Loading cached scaler from {scaler_path}")
        return joblib.load(scaler_path)

    logger.info("Fitting StandardScaler on training data (per column)")
    scaler = StandardScaler()
    scaler.fit(train_split.X)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    return scaler


def apply_scaler(scaler: StandardScaler, split: AI4PainSplit) -> AI4PainSplit:
    """Return a new split with X transformed by the fitted scaler."""
    X_scaled = scaler.transform(split.X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return AI4PainSplit(
        X=X_scaled,
        y=split.y,
        subject_ids=split.subject_ids,
        trial_ids=split.trial_ids,
        feature_names=split.feature_names,
    )


# ----------------------------------------------------------------------------
# Class weight helper
# ----------------------------------------------------------------------------


def compute_class_weights(y: np.ndarray) -> dict:
    """Return sklearn-style ``class_weight='balanced'`` dict from labels."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}
