#!/usr/bin/env bash
#SBATCH --job-name=evolve-submission
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Single-task SLURM script for a HIP-G test-set submission. Mirrors
# run_array.slurm's environment setup (venv .venv, Python 3.11, dependency
# check) so the submission runs in the SAME environment as every iteration
# job. The submission trains a model + predicts the blinded test split;
# it must NOT run on a login node (wrong Python, no GPU, login-node compute).
#
# Submit from the EVOLVE project root on Explorer:
#   cd /projects/SensingandInnovationLab/vignankamarthi/EVOLVE
#   sbatch --export=ALL,SUBMISSION_DIR=experiments/submission_01 \
#          scripts/run_submission.slurm
#
# Reusable for submissions 1-5: point SUBMISSION_DIR at the right dir.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/EVOLVE"

if [[ -z "${SUBMISSION_DIR:-}" ]]; then
    echo "FATAL: SUBMISSION_DIR env var is required (path to experiments/submission_NN)" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

if [[ ! -d "${SUBMISSION_DIR}" || ! -f "${SUBMISSION_DIR}/run.py" ]]; then
    echo "FATAL: ${SUBMISSION_DIR}/run.py not found" >&2
    exit 1
fi

# Activate venv (created locally with ~/python/bin/python3.11 -m venv .venv).
# Identical to run_array.slurm -- the submission runs in the SAME environment.
source .venv/bin/activate

# Self-healing dependency check (mirrors run_array.slurm).
check_dep() {
    local import_name="$1"
    local install_spec="$2"
    python -c "import ${import_name}" 2>/dev/null || {
        echo "Installing missing dependency: ${install_spec}"
        pip install ${install_spec} || {
            echo "FATAL: pip install ${install_spec} failed"
            exit 1
        }
    }
}

check_dep numpy "numpy>=1.26"
check_dep pandas "pandas>=2.1"
check_dep scipy "scipy>=1.12"
check_dep sklearn "scikit-learn>=1.4"
check_dep pyarrow "pyarrow>=15.0"
check_dep tqdm "tqdm>=4.66"
check_dep torch "torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"

# Verify CUDA. Warn but do not kill (CPU fallback works, just slow).
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" != "True" ]]; then
    echo "WARNING: CUDA unavailable on $(hostname). Run will use CPU (slower)."
else
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo "Submission: running ${SUBMISSION_DIR}/run.py at $(date) on $(hostname)"

python -u "${SUBMISSION_DIR}/run.py" --run-dir "${SUBMISSION_DIR}"

echo "Submission ${SUBMISSION_DIR} complete at $(date)"
