#!/usr/bin/env bash
# One-time data sync from Mac to Explorer.
#
# Data is private challenge material. NEVER committed to git (data/raw/ is
# gitignored). Sync via rsync only. Run this manually once after HIP-A
# (data download). Re-run only if the data changes upstream.
#
# Usage:
#   scripts/sync_data_to_cluster.sh

set -euo pipefail

EXPLORER_USER="${EXPLORER_USER:-kamarthi.v}"
# Use the transfer node (xfer.discovery.neu.edu) for rsync, not the login node.
EXPLORER_HOST="${EXPLORER_HOST:-xfer.discovery.neu.edu}"
EXPLORER_REPO="${EXPLORER_REPO:-/projects/SensingandInnovationLab/vignankamarthi/EVOLVE}"

LOCAL_DIR="data/raw/"
REMOTE_DIR="${EXPLORER_USER}@${EXPLORER_HOST}:${EXPLORER_REPO}/data/raw/"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "no such directory: ${LOCAL_DIR}" >&2
  exit 1
fi

echo "syncing data/raw/ to ${REMOTE_DIR}"
echo "size: $(du -sh "${LOCAL_DIR}" | cut -f1)"

# Ensure remote parent dir exists (macOS rsync 2.6.9 has no --mkpath).
ssh "${EXPLORER_USER}@${EXPLORER_HOST}" "mkdir -p '${EXPLORER_REPO}/data/raw'"

rsync -avz --partial --progress "${LOCAL_DIR}" "${REMOTE_DIR}"
echo "data sync complete"
