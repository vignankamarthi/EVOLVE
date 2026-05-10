#!/usr/bin/env bash
#SBATCH --job-name=evolve-iter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Manual SLURM template. ANTIPATTERNS rule 10: framework Python NEVER calls
# sbatch. Vignan submits this manually at HIP-E.
#
# Submit from the EVOLVE project root on Explorer:
#   cd /projects/SensingandInnovationLab/vignankamarthi/EVOLVE
#   sbatch --export=ALL,RUN_ID=baseline_bigru scripts/run_experiment.slurm
#
# H200 specifically (--gres=gpu:h200:1) avoids A100 (known data leakage).

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/EVOLVE"

if [[ -z "${RUN_ID:-}" ]]; then
    echo "FATAL: RUN_ID env var is required. Use: sbatch --export=ALL,RUN_ID=<id> ..." >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

RUN_DIR="experiments/${RUN_ID}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "FATAL: no such run directory: ${PROJECT_DIR}/${RUN_DIR}" >&2
    exit 1
fi

# Activate venv (created locally with ~/python/bin/python3.11 -m venv .venv)
source .venv/bin/activate

# Self-healing dependency check. Per NEU-HPC convention.
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
# CRITICAL: torch must be 2.5.1+cu121 for Explorer H200 (CUDA 12.0 driver).
# Latest torch (2.11+) requires CUDA 13 and silently falls back to CPU.
check_dep torch "torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"

# Verify CUDA. Warn but don't kill (CPU fallback works, just slow).
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" != "True" ]]; then
    echo "WARNING: CUDA unavailable on $(hostname). Run will use CPU (10x slower)."
else
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "torch CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
fi

echo "Running ${RUN_DIR}/run.py at $(date) on $(hostname)"

python -u "${RUN_DIR}/run.py" --run-dir "${RUN_DIR}"

echo "run ${RUN_ID} complete at $(date), result at ${RUN_DIR}/result.json"
