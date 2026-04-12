"""Classical ML training entry point for AI4Pain 2026.

Adapted from `Blood-Pressure-Inference-with-BVP/scripts/train_models.py`.
Three execution modes:

  1. ``--dry-run``: synthetic data only, runs the full pipeline (data load,
     scale, optional Optuna HPO, train, evaluate, leaderboard) end-to-end
     with a small synthetic dataset that respects the subject-level split.
     This is the Phase 2 verification target: it must run cleanly without any
     real data.

  2. Real data, no tuning: load Rust feature CSVs joined per ablation config,
     train with default params (or saved best_params if present), evaluate
     on the validation split, write leaderboard.

  3. Real data with ``--tune``: same as (2) but runs Optuna HPO on the
     training split first.

Usage examples:
    python scripts/train_models.py --config bvp_eda --dry-run
    python scripts/train_models.py --config bvp_eda
    python scripts/train_models.py --config bvp_eda --tune --n-trials 100
    python scripts/train_models.py --config all_four --tune --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/train_models.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (  # noqa: E402
    AI4PainSplit,
    apply_scaler,
    fit_scaler_on_train,
    load_split,
    make_synthetic_split,
)
from src.evaluation import evaluate_model, generate_leaderboard  # noqa: E402
from src.models import load_status, train_all  # noqa: E402
from src.tuning import tune_all  # noqa: E402
from src.utils import setup_logging, timer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_CONFIG_PATH = REPO_ROOT / "configs" / "ablation_configs.json"
DEFAULT_FEATURES_DIR = REPO_ROOT / "data" / "features"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "raw" / "labels.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "classical_ml"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "classical_ml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        choices=["bvp_eda", "bvp_eda_resp", "all_four"],
        help="Ablation configuration (which signals to use)",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory holding the Rust extractor outputs (results_{split}_{signal}.csv)",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Label metadata CSV (file_name, subject_id, label) for joining",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Base checkpoint directory (namespaced by config)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base results directory (namespaced by config)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logistic_regression,rf,xgb,lgbm",
        help="Comma-separated model names",
    )
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO before training")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per model")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if present",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing checkpoints before starting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data instead of real Rust extractor outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_ablation_config(name: str) -> dict:
    with open(ABLATION_CONFIG_PATH) as f:
        all_configs = json.load(f)
    return all_configs[name]


def get_real_splits(args, ablation: dict, logger) -> tuple[AI4PainSplit, AI4PainSplit]:
    signals = ablation["signals"]
    logger.info(f"Loading real features for ablation '{args.config}' (signals: {signals})")
    train_split = load_split(args.features_dir, args.labels_path, "train", signals)
    val_split = load_split(args.features_dir, args.labels_path, "validation", signals)
    return train_split, val_split


def get_synthetic_splits(ablation: dict, seed: int, logger) -> tuple[AI4PainSplit, AI4PainSplit]:
    n_features = ablation["n_features"]
    logger.info(
        f"DRY-RUN: synthesizing data with {n_features} features "
        f"(20 train subjects x 6 trials, 8 val subjects x 6 trials)"
    )
    train = make_synthetic_split(n_subjects=20, n_trials_per_subject=6, n_features=n_features, seed=seed)
    val = make_synthetic_split(
        n_subjects=8, n_trials_per_subject=6, n_features=n_features, seed=seed + 1
    )
    return train, val


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logging(name="ai4pain_2026")

    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh")
        return 1

    ablation = load_ablation_config(args.config)
    checkpoint_dir = args.checkpoint_dir / args.config
    results_dir = args.results_dir / args.config
    models_to_train = [m.strip() for m in args.models.split(",") if m.strip()]

    logger.info(f"Config: {args.config}")
    logger.info(f"Models: {models_to_train}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Mode: {'DRY-RUN (synthetic)' if args.dry_run else 'real data'}")

    if not args.resume and not args.fresh and checkpoint_dir.exists():
        status = load_status(checkpoint_dir)
        if status.get("completed"):
            logger.error(
                "Checkpoints exist. Use --resume to continue or --fresh to restart."
            )
            return 1

    if args.fresh and checkpoint_dir.exists():
        import shutil

        logger.warning("Deleting existing checkpoints (--fresh)")
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    with timer("Loading data", logger):
        if args.dry_run:
            train_split, val_split = get_synthetic_splits(ablation, args.seed, logger)
        else:
            train_split, val_split = get_real_splits(args, ablation, logger)

    logger.info(
        f"Train: {train_split.n_samples} samples, {train_split.n_features} features, "
        f"class dist {train_split.class_distribution}"
    )
    logger.info(
        f"Val:   {val_split.n_samples} samples, {val_split.n_features} features, "
        f"class dist {val_split.class_distribution}"
    )

    # ------------------------------------------------------------------
    # 2. Scale (fit on train only)
    # ------------------------------------------------------------------
    with timer("Fitting StandardScaler", logger):
        scaler = fit_scaler_on_train(train_split, checkpoint_dir, force_refit=args.fresh)
        train_scaled = apply_scaler(scaler, train_split)
        val_scaled = apply_scaler(scaler, val_split)

    # ------------------------------------------------------------------
    # 3. Hyperparameter tuning (optional)
    # ------------------------------------------------------------------
    if args.tune:
        with timer("Hyperparameter tuning", logger):
            best_params = tune_all(
                train_scaled.X,
                train_scaled.y,
                train_scaled.subject_ids,
                checkpoint_dir,
                models_to_tune=models_to_train,
                n_trials=args.n_trials,
            )
    else:
        best_params = {}
        params_dir = checkpoint_dir / "best_params"
        if params_dir.exists():
            for f in params_dir.glob("*.json"):
                with open(f) as fp:
                    saved = json.load(fp)
                best_params[saved["model"]] = saved["best_params"]
            logger.info(f"Loaded {len(best_params)} saved parameter sets")
        if not best_params:
            logger.info("No saved params, using defaults from configs/model_configs.json")
            with open(REPO_ROOT / "configs" / "model_configs.json") as fp:
                model_defaults = json.load(fp)
            for m in models_to_train:
                best_params[m] = model_defaults.get(m, {})

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    with timer("Model training", logger):
        trained = train_all(
            train_scaled.X,
            train_scaled.y,
            best_params,
            checkpoint_dir,
            models_to_train=models_to_train,
            resume=args.resume,
        )

    # ------------------------------------------------------------------
    # 5. Evaluate on validation split
    # ------------------------------------------------------------------
    with timer("Validation evaluation", logger):
        for model_name, model in trained.items():
            y_pred = model.predict(val_scaled.X)
            try:
                y_proba = model.predict_proba(val_scaled.X)
            except AttributeError:
                y_proba = None
            evaluate_model(model_name, val_scaled.y, y_pred, y_proba, results_dir)

    # ------------------------------------------------------------------
    # 6. Leaderboard
    # ------------------------------------------------------------------
    leaderboard = generate_leaderboard(results_dir)
    logger.info("=" * 60)
    logger.info(f"LEADERBOARD ({args.config})")
    logger.info("=" * 60)
    for entry in leaderboard["entries"]:
        flag = " [COLLAPSED]" if entry["single_class_collapse"] else ""
        logger.info(
            f"  {entry['model']:22s}: bal_acc={entry['balanced_accuracy']:.4f}, "
            f"macro_f1={entry['macro_f1']:.4f}, AUC-OVR={entry['auc_ovr']:.4f}{flag}"
        )

    logger.info(f"Done. Best model: {leaderboard.get('best_model')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
