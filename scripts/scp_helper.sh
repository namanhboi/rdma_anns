#!/bin/bash
set -euo pipefail
ALL_CLOUDLAB_HOSTS=(
    "namanh@er039.utah.cloudlab.us"
    "namanh@er082.utah.cloudlab.us"
    "namanh@er076.utah.cloudlab.us"
    "namanh@er050.utah.cloudlab.us"
    "namanh@er104.utah.cloudlab.us"
    "namanh@er124.utah.cloudlab.us"
    "namanh@er001.utah.cloudlab.us"
    "namanh@er040.utah.cloudlab.us"
    "namanh@er032.utah.cloudlab.us"
    "namanh@er126.utah.cloudlab.us"					
)
REMOTE_WORKDIR="/users/namanh/workspace/rdma_anns/"
SSH_OPTS="-o StrictHostKeyChecking=no -o ForwardAgent=no -o LogLevel=ERROR -r"

# Check arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: ${BASH_SOURCE[0]} <experiment_name> <server_id>"
    echo "  experiment_name"
    echo "  server_id: 5, 10"
    exit 1
fi

EXPERIMENT_NAME=$1
SERVER_ID=$2

# Extract last 2 directory components
# EXPERIMENT_NAME=$(echo "$FULL_LOCAL_LOG_PATH" | rev | cut -d'/' -f1-2 | rev)
echo "Experiment name: ${EXPERIMENT_NAME}"

REMOTE_LOG_DIR="${REMOTE_WORKDIR}/logs/${EXPERIMENT_NAME}"
LOCAL_LOG_DIR="${HOME}/workspace/rdma_anns/logs/${EXPERIMENT_NAME}"
LAST_CLOUDLAB_HOST="${ALL_CLOUDLAB_HOSTS[$SERVER_ID]}"

# Copy results
scp $SSH_OPTS "$LAST_CLOUDLAB_HOST:${REMOTE_LOG_DIR}" "$LOCAL_LOG_DIR" && {
    echo "    ✓ results copied to: $LOCAL_LOG_DIR"
} || {
    echo "    ⚠ Could not copy results from $LAST_CLOUDLAB_HOST"
}
