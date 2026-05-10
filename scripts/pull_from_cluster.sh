#!/usr/bin/env bash
# Manual cluster pull helper. ANTIPATTERNS rule 10: framework Python NEVER
# calls this. Vignan runs by hand at HIP-F.
#
# Usage: scripts/pull_from_cluster.sh <run_id>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
EXPLORER_USER="${EXPLORER_USER:-kamarthi.v}"
# Use the transfer node (xfer.discovery.neu.edu) for rsync, not the login node.
EXPLORER_HOST="${EXPLORER_HOST:-xfer.discovery.neu.edu}"
EXPLORER_REPO="${EXPLORER_REPO:-/projects/SensingandInnovationLab/vignankamarthi/EVOLVE}"

REMOTE_DIR="${EXPLORER_USER}@${EXPLORER_HOST}:${EXPLORER_REPO}/experiments/${RUN_ID}/"
LOCAL_DIR="experiments/${RUN_ID}/"

mkdir -p "${LOCAL_DIR}"
rsync -avz --partial --progress "${REMOTE_DIR}" "${LOCAL_DIR}"
echo "pulled ${RUN_ID} from ${REMOTE_DIR}"
