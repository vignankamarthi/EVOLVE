"""Classical ML model training with checkpointing for AI4Pain 2026.

Adapted from `Blood-Pressure-Inference-with-BVP/src/models.py`. The BP
inference version trained 5 regressors (Ridge / DT / RF / XGB / LGBM) for two
targets (SBP, DBP). This version trains 4 classifiers
(LogisticRegression / RF / XGB / LGBM) for a single target (3-class pain
localization). The checkpointing infrastructure is unchanged: warm_start RF
every 50 trees, XGBoost incremental rounds every 25, LightGBM checkpoint
callback every 25 iterations, status.json for the orchestrator-level resume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# NOTE: xgboost and lightgbm are imported LAZILY inside the functions that
# need them. On macOS they bundle their own libomp which conflicts with
# PyTorch's libomp when both are imported in the same process, leading to
# segfaults. Lazy import avoids triggering the conflict for code paths that
# only need sklearn (e.g. the DL smoke tests in tests/test_scaffolding.py).

from .utils import atomic_json_write, clear_memory, get_logger, load_json


logger = get_logger("ai4pain_2026")


# Single target for AI4Pain 2026: pain localization class label
TARGET = "pain_class"


# ----------------------------------------------------------------------------
# Status / progress helpers
# ----------------------------------------------------------------------------


def load_status(checkpoint_dir: Path) -> Dict:
    """Load training status checkpoint."""
    return load_json(checkpoint_dir / "status.json") or {"completed": [], "in_progress": None}


def save_status(checkpoint_dir: Path, status: Dict) -> None:
    """Persist training status atomically."""
    atomic_json_write(checkpoint_dir / "status.json", status)


def is_model_done(checkpoint_dir: Path, model_name: str) -> bool:
    """Check whether a final model file exists and the orchestrator marked it done."""
    model_path = checkpoint_dir / "models" / f"{model_name}.pkl"
    status = load_status(checkpoint_dir)
    return model_path.exists() and model_name in status.get("completed", [])


# ----------------------------------------------------------------------------
# Per-model training functions
# ----------------------------------------------------------------------------


def train_logistic_regression(
    X_train, y_train, params: Dict, checkpoint_dir: Path
) -> LogisticRegression:
    """Linear baseline. Fast, no mid-training checkpointing needed."""
    logger.info("Training LogisticRegression (pain_class)...")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    out = checkpoint_dir / "models" / "logistic_regression.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    logger.info(f"  Saved to {out}")
    return model


def train_random_forest(
    X_train, y_train, params: Dict, checkpoint_dir: Path
) -> RandomForestClassifier:
    """Random Forest with warm_start checkpointing every 50 trees.

    Resume loads the latest checkpoint and continues from there. Identical
    pattern to the BP inference version.
    """
    logger.info("Training RandomForestClassifier...")
    models_dir = checkpoint_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    target_trees = params.get("n_estimators", 500)
    step = 50

    progress_path = checkpoint_dir / "rf_progress.json"
    progress = load_json(progress_path) or {"current_trees": 0}
    start_trees = progress["current_trees"]

    model: Optional[RandomForestClassifier] = None
    if start_trees > 0:
        latest = models_dir / f"rf_n{start_trees}.pkl"
        if latest.exists():
            logger.info(f"  Resuming from {start_trees} trees")
            model = joblib.load(latest)
        else:
            logger.info(
                f"  Progress file says {start_trees} trees but no model artifact, restarting"
            )
            start_trees = 0

    current = start_trees
    while current < target_trees:
        next_target = min(current + step, target_trees)
        if model is None:
            kwargs = {k: v for k, v in params.items() if k != "n_estimators"}
            model = RandomForestClassifier(
                n_estimators=next_target,
                warm_start=True,
                random_state=42,
                n_jobs=-1,
                **kwargs,
            )
        else:
            model.n_estimators = next_target
        model.fit(X_train, y_train)
        current = next_target

        ckpt_path = models_dir / f"rf_n{current}.pkl"
        joblib.dump(model, ckpt_path)
        atomic_json_write(progress_path, {"current_trees": current, "target_trees": target_trees})
        logger.info(f"  RF checkpoint: {current}/{target_trees} trees")

    final_path = models_dir / "rf.pkl"
    joblib.dump(model, final_path)
    logger.info(f"  Final RF saved to {final_path}")
    return model  # type: ignore[return-value]


def train_xgboost(X_train, y_train, params: Dict, checkpoint_dir: Path):
    """XGBoost classifier with incremental round-based checkpointing every 25 rounds."""
    import xgboost as xgb  # lazy import (libomp conflict on macOS)

    logger.info("Training XGBoostClassifier...")
    models_dir = checkpoint_dir / "models"
    xgb_ckpt_dir = checkpoint_dir / "xgb_pain_class"
    xgb_ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    params = dict(params)
    n_rounds = params.pop("n_estimators", 500)

    progress_path = checkpoint_dir / "xgb_progress.json"
    progress = load_json(progress_path) or {"current_round": 0}
    start_round = progress["current_round"]

    if start_round >= n_rounds:
        final = models_dir / "xgb.pkl"
        if final.exists():
            return joblib.load(final)

    step = 25
    current = start_round
    model = None  # xgb.XGBClassifier; type left dynamic for lazy import

    while current < n_rounds:
        next_target = min(current + step, n_rounds)
        batch_size = next_target - current

        # Force multi-class softprob; remove user-supplied conflict
        local_params = dict(params)
        local_params.setdefault("objective", "multi:softprob")
        local_params.setdefault("num_class", 3)
        # XGBClassifier infers num_class from y; objective passed through

        model = xgb.XGBClassifier(
            n_estimators=batch_size,
            random_state=42,
            n_jobs=-1,
            **local_params,
        )
        model.fit(X_train, y_train)
        current = next_target

        ckpt_path = xgb_ckpt_dir / f"xgb_round_{current}.json"
        model.get_booster().save_model(str(ckpt_path))
        atomic_json_write(progress_path, {"current_round": current, "target_rounds": n_rounds})
        logger.info(f"  XGBoost checkpoint: {current}/{n_rounds} rounds")

    final_path = models_dir / "xgb.pkl"
    joblib.dump(model, final_path)
    logger.info(f"  Final XGBoost saved to {final_path}")
    return model  # type: ignore[return-value]


class LGBMCheckpointCallback:
    """LightGBM callback that checkpoints the booster every ``interval`` iterations."""

    def __init__(self, checkpoint_dir: Path, interval: int = 25):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, env):  # noqa: ANN001 (env is LightGBM's CallbackEnv)
        iteration = env.iteration + 1
        if iteration % self.interval == 0:
            path = self.checkpoint_dir / f"lgbm_iter_{iteration}.txt"
            env.model.save_model(str(path))
            progress_path = self.checkpoint_dir.parent / "lgbm_progress.json"
            atomic_json_write(
                progress_path,
                {"current_iter": iteration, "target_iter": env.end_iteration},
            )


def train_lightgbm(X_train, y_train, params: Dict, checkpoint_dir: Path):
    """LightGBM classifier with custom checkpoint callback every 25 iterations."""
    import lightgbm as lgb  # lazy import (libomp conflict on macOS)

    logger.info("Training LightGBMClassifier...")
    models_dir = checkpoint_dir / "models"
    lgbm_ckpt_dir = checkpoint_dir / "lgbm_pain_class"
    lgbm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    params = dict(params)
    n_estimators = params.pop("n_estimators", 500)
    params.setdefault("objective", "multiclass")
    params.setdefault("num_class", 3)

    progress_path = checkpoint_dir / "lgbm_progress.json"
    progress = load_json(progress_path) or {"current_iter": 0}
    start_iter = progress["current_iter"]
    remaining = n_estimators - start_iter

    init_model = None
    if start_iter > 0 and remaining > 0:
        ckpt_files = sorted(lgbm_ckpt_dir.glob("lgbm_iter_*.txt"))
        if ckpt_files:
            init_model = str(ckpt_files[-1])
            logger.info(f"  Resuming LightGBM from iteration {start_iter}")

    if remaining <= 0:
        final = models_dir / "lgbm.pkl"
        if final.exists():
            return joblib.load(final)

    callback = LGBMCheckpointCallback(lgbm_ckpt_dir, interval=25)

    model = lgb.LGBMClassifier(
        n_estimators=remaining,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        X_train, y_train,
        init_model=init_model,
        callbacks=[callback],
    )

    final_path = models_dir / "lgbm.pkl"
    joblib.dump(model, final_path)
    atomic_json_write(progress_path, {"current_iter": n_estimators, "target_iter": n_estimators})
    logger.info(f"  Final LightGBM saved to {final_path}")
    return model


# ----------------------------------------------------------------------------
# Model registry and orchestration
# ----------------------------------------------------------------------------


MODELS = {
    "logistic_regression": train_logistic_regression,
    "rf": train_random_forest,
    "xgb": train_xgboost,
    "lgbm": train_lightgbm,
}


def train_all(
    X_train,
    y_train,
    best_params: Dict,
    checkpoint_dir: Path,
    models_to_train: Optional[list] = None,
    resume: bool = False,
) -> Dict:
    """Train every model in ``models_to_train`` for the single pain_class target.

    Returns ``{model_name: trained_model}``. Re-uses existing models on disk if
    ``resume=True`` and the orchestrator status marks them complete.
    """
    if models_to_train is None:
        models_to_train = list(MODELS.keys())

    trained: Dict[str, object] = {}
    status = load_status(checkpoint_dir)

    for model_name in models_to_train:
        if model_name not in MODELS:
            logger.warning(f"Unknown model: {model_name}")
            continue

        if resume and is_model_done(checkpoint_dir, model_name):
            logger.info(f"SKIP {model_name} (already completed)")
            trained[model_name] = joblib.load(checkpoint_dir / "models" / f"{model_name}.pkl")
            continue

        status["in_progress"] = model_name
        save_status(checkpoint_dir, status)

        params = best_params.get(model_name, {}).copy()
        try:
            model = MODELS[model_name](X_train, y_train, params, checkpoint_dir)
            trained[model_name] = model
            if model_name not in status["completed"]:
                status["completed"].append(model_name)
            status["in_progress"] = None
            save_status(checkpoint_dir, status)
        except Exception as exc:
            logger.error(f"FAILED {model_name}: {exc}")
            status["in_progress"] = None
            save_status(checkpoint_dir, status)
            raise

        clear_memory()

    return trained
