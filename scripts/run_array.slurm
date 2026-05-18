#!/usr/bin/env bash
#SBATCH --job-name=evolve-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Manual SLURM array template. ANTIPATTERNS rule 10: framework Python NEVER
# calls sbatch. Vignan submits this manually.
#
# Submit from the EVOLVE project root on Explorer with a manifest path:
#   cd /projects/SensingandInnovationLab/vignankamarthi/EVOLVE
#   sbatch --array=0-14%4 --export=ALL,MANIFEST=experiments/subset_transfer/manifest.json \
#          scripts/run_array.slurm
#
# --array=0-14%4 means: 15 task ids (0..14), max 4 concurrent (cluster cap).
# Adjust the upper bound and concurrency for other array sizes.
#
# Each array task reads ${MANIFEST}, looks up experiments[SLURM_ARRAY_TASK_ID],
# and runs that experiment dir's run.py. H200 explicit avoids A100 leakage.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/EVOLVE"

if [[ -z "${MANIFEST:-}" ]]; then
    echo "FATAL: MANIFEST env var is required (path to manifest.json)" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

if [[ ! -f "${MANIFEST}" ]]; then
    echo "FATAL: manifest not found at ${PROJECT_DIR}/${MANIFEST}" >&2
    exit 1
fi

# QOS guard: NEU Explorer caps submitted jobs per user at 8. Mac-side
# generation must keep manifests at <= 8 entries or this slurm script
# will be invoked with --array indices that fail submission.
N_TASKS=$(python -c "import json; print(len(json.load(open('${MANIFEST}'))['experiments']))" 2>/dev/null || echo "?")
if [[ "$N_TASKS" != "?" && "$N_TASKS" -gt 8 ]]; then
    echo "WARNING: manifest has ${N_TASKS} entries; cluster QOS cap is 8." >&2
    echo "         If sbatch submission failed, cap the manifest at 8 and resubmit." >&2
fi

# Activate venv (created locally with ~/python/bin/python3.11 -m venv .venv)
source .venv/bin/activate

# Self-healing dependency check.
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

# Look up this task's RUN_ID from the manifest.
RUN_ID=$(python -c "
import json, sys
m = json.load(open('${MANIFEST}'))
print(m['experiments'][${SLURM_ARRAY_TASK_ID}]['run_id'])
")
RUN_DIR="$(python -c "
import json, sys
m = json.load(open('${MANIFEST}'))
print(m['experiments'][${SLURM_ARRAY_TASK_ID}]['run_dir'])
")"

if [[ -z "${RUN_ID}" || ! -d "${RUN_DIR}" ]]; then
    echo "FATAL: invalid array index ${SLURM_ARRAY_TASK_ID}. RUN_ID=${RUN_ID} RUN_DIR=${RUN_DIR}" >&2
    exit 1
fi

# Re-run guard (2026-05-11): refuse to overwrite an existing result.json.
# Prevents the iter_0001 / iter_0002 mix-up. To re-run, delete the stale
# result.json first OR adjust the manifest to point at a fresh run_dir.
if [[ -f "${RUN_DIR}/result.json" ]]; then
    echo "FATAL: ${RUN_DIR}/result.json already exists." >&2
    echo "This task was already trained. Delete the result.json or edit the manifest." >&2
    exit 1
fi

# Verify CUDA. Warn but don't kill (CPU fallback works, just slow).
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" != "True" ]]; then
    echo "WARNING: CUDA unavailable on $(hostname). Run will use CPU (10x slower)."
else
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: running ${RUN_DIR}/run.py at $(date) on $(hostname)"

python -u "${RUN_DIR}/run.py" --run-dir "${RUN_DIR}"

echo "Array task ${SLURM_ARRAY_TASK_ID} (${RUN_ID}) complete at $(date)"
