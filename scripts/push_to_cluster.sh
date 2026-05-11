#!/usr/bin/env bash
# Manual cluster push helper. ANTIPATTERNS rule 10: framework Python NEVER
# calls this. Vignan runs by hand at HIP-D.
#
# Usage: scripts/push_to_cluster.sh <run_id>
#
# Pushes experiments/<run_id>/ to Explorer. Adjust EXPLORER_USER and
# EXPLORER_HOST and EXPLORER_REPO to your cluster.

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

LOCAL_DIR="experiments/${RUN_ID}/"
REMOTE_DIR="${EXPLORER_USER}@${EXPLORER_HOST}:${EXPLORER_REPO}/experiments/${RUN_ID}/"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "no such directory: ${LOCAL_DIR}" >&2
  exit 1
fi

# Ensure the remote parent directory exists. macOS ships rsync 2.6.9 which
# lacks --mkpath, so we create it via an explicit ssh hop before rsync.
ssh "${EXPLORER_USER}@${EXPLORER_HOST}" "mkdir -p '${EXPLORER_REPO}/experiments/${RUN_ID}'"

rsync -avz --partial --progress "${LOCAL_DIR}" "${REMOTE_DIR}"
echo "pushed ${RUN_ID} to ${REMOTE_DIR}"
