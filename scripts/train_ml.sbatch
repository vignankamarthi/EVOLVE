#!/bin/bash
#SBATCH --job-name=ai4pain-train-ml
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_ml_%j.out
#SBATCH --error=logs/train_ml_%j.err

# AI4Pain 2026 classical ML HPO + training (Phase 2 complete).
#
# Required env vars:
#   CONFIG -- one of bvp_eda, bvp_eda_resp, all_four
#
# Optional:
#   MODELS    -- comma-separated subset (default: logistic_regression,rf,xgb,lgbm)
#   N_TRIALS  -- Optuna trials per model (default: 100)
#   RESUME    -- set to "--resume" to continue from checkpoints
#   FRESH     -- set to "--fresh" to delete checkpoints first

set -euo pipefail

: "${CONFIG:?CONFIG is required (bvp_eda|bvp_eda_resp|all_four)}"
: "${MODELS:=logistic_regression,rf,xgb,lgbm}"
: "${N_TRIALS:=100}"
: "${RESUME:=}"
: "${FRESH:=}"

mkdir -p logs

python scripts/train_models.py \
    --config "${CONFIG}" \
    --models "${MODELS}" \
    --tune \
    --n-trials "${N_TRIALS}" \
    ${RESUME} ${FRESH}

echo "[train_ml] Done: ${CONFIG} (HPO + train + eval)"
