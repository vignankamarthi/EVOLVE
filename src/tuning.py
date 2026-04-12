"""Optuna hyperparameter tuning for AI4Pain 2026 classifiers.

Adapted from `Blood-Pressure-Inference-with-BVP/src/tuning.py`. Two key
changes from BP inference:

  1. Optimization direction flipped from ``minimize`` (MAE) to ``maximize``
     (balanced accuracy).
  2. Cross-validation uses ``StratifiedGroupKFold`` so each fold is both
     class-stratified AND subject-disjoint. This prevents the silent
     leakage that would happen with vanilla ``StratifiedKFold`` if a single
     subject's trials landed in both train and validation folds.

The four search spaces match the model lineup in ``models.py``:
LogisticRegression, RandomForestClassifier, XGBoostClassifier,
LightGBMClassifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# xgboost and lightgbm imported LAZILY in create_model() to avoid the macOS
# libomp conflict with PyTorch (see src/models.py for the same pattern).

from .utils import atomic_json_write, get_logger


logger = get_logger("ai4pain_2026")
optuna.logging.set_verbosity(optuna.logging.WARNING)


RANDOM_SEED = 42
N_TRIALS = 100
CV_FOLDS = 5
PRIMARY_METRIC = "balanced_accuracy"
STUDY_DIRECTION = "maximize"


# ----------------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------------


def get_search_space(model_name: str, trial: optuna.Trial) -> Dict:
    """Define the Optuna search space for one model."""
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
        }
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": 3,
        }
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
        }
    raise ValueError(f"Unknown model: {model_name}")


def create_model(model_name: str, params: Dict):
    """Instantiate a classifier with the given hyperparameters.

    xgboost and lightgbm are imported lazily here to avoid the macOS libomp
    conflict with PyTorch when only the LR / RF code path is exercised.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "rf":
        return RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1)
    if model_name == "xgb":
        import xgboost as xgb  # lazy import

        return xgb.XGBClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1)
    if model_name == "lgbm":
        import lightgbm as lgb  # lazy import

        return lgb.LGBMClassifier(
            **params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
        )
    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------------------------------------------------------
# CV scoring
# ----------------------------------------------------------------------------


def stratified_group_cv_score(
    model_name: str,
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> float:
    """Run subject-disjoint stratified CV and return mean balanced accuracy.

    Uses ``StratifiedGroupKFold`` to keep each subject in only one fold while
    still balancing the class distribution across folds. This is the right
    inner-CV scheme for AI4Pain because the train split has only 41 subjects
    and naive StratifiedKFold would put trials from the same subject on both
    sides of a fold boundary.

    Returns
    -------
    float
        Mean balanced accuracy across the folds. Returns 0.0 if a fold raises
        (defensive: prevents Optuna from crashing on a degenerate trial).
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = create_model(model_name, params)
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, y_pred))
        except Exception as exc:
            logger.warning(f"  CV fold raised in {model_name}: {exc}")
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------------
# Per-model tuning entry
# ----------------------------------------------------------------------------


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_ids: np.ndarray,
    checkpoint_dir: Path,
    n_trials: int = N_TRIALS,
) -> Dict:
    """Run Optuna tuning for one classifier on the training split.

    SQLite-backed study persists across SLURM job restarts. Best params are
    written to ``checkpoint_dir/best_params/{model_name}.json``.
    """
    storage_path = checkpoint_dir / f"optuna_{model_name}.db"
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"ai4pain_{model_name}"

    logger.info(f"Tuning {model_name}: {n_trials} trials")
    logger.info(f"  Storage: {storage_path}")

    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(model_name, trial)
        return stratified_group_cv_score(
            model_name, params, X_train, y_train, subject_ids
        )

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=STUDY_DIRECTION,
        sampler=sampler,
        load_if_exists=True,
    )

    completed = len(study.trials)
    remaining = n_trials - completed
    if remaining <= 0:
        logger.info(f"  Already completed {completed} trials, skipping")
    else:
        logger.info(f"  {completed} trials done, running {remaining} more")
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_value = float(study.best_value)

    out = checkpoint_dir / "best_params" / f"{model_name}.json"
    atomic_json_write(
        out,
        {
            "model": model_name,
            "best_params": best_params,
            "best_balanced_accuracy": best_value,
            "n_trials": len(study.trials),
        },
    )
    logger.info(f"  Best balanced accuracy: {best_value:.4f}")
    logger.info(f"  Best params: {best_params}")
    return best_params


def tune_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_ids: np.ndarray,
    checkpoint_dir: Path,
    models_to_tune: Optional[list] = None,
    n_trials: int = N_TRIALS,
) -> Dict:
    """Tune every model in ``models_to_tune``. Returns ``{model_name: best_params}``."""
    if models_to_tune is None:
        models_to_tune = ["logistic_regression", "rf", "xgb", "lgbm"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_params: Dict[str, Dict] = {}

    for model_name in models_to_tune:
        params_path = checkpoint_dir / "best_params" / f"{model_name}.json"
        if params_path.exists():
            import json

            with open(params_path) as f:
                saved = json.load(f)
            if saved.get("n_trials", 0) >= n_trials:
                logger.info(
                    f"SKIP tuning {model_name} (already have {saved['n_trials']} trials)"
                )
                all_params[model_name] = saved["best_params"]
                continue
        all_params[model_name] = tune_model(
            model_name, X_train, y_train, subject_ids, checkpoint_dir, n_trials
        )

    return all_params
