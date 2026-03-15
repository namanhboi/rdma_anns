#!/usr/bin/bash

# Exit on error, undefined vars, and pipe failures
set -euo pipefail

# Get the directory where the script lives
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load your host array (assumes ALL_CLOUDLAB_HOSTS is defined there)
if [[ -f "${SCRIPT_DIR}/cloudlab_addresses.sh" ]]; then
    source "${SCRIPT_DIR}/cloudlab_addresses.sh"
else
    echo "Error: cloudlab_addresses.sh not found."
    exit 1
fi

# Ensure at least two arguments: file and at least one node ID
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file_to_transfer> <node_id1> [node_id2 ...]"
    exit 1
fi

FILE=$1
shift 1  # The remaining arguments ($@) are now just the Node IDs

for NODE_ID in "$@"; do
    # Check if the ID actually exists in your sourced array
    if [[ -v "ALL_CLOUDLAB_HOSTS[$NODE_ID]" ]]; then
        CLOUDLAB_HOST=${ALL_CLOUDLAB_HOSTS[$NODE_ID]}
        
        echo "--- Syncing to Node $NODE_ID ($CLOUDLAB_HOST) ---"
        # -v: verbose, -L: follow symlinks, -p: preserve permissions
        rsync -vL "$FILE" "${CLOUDLAB_HOST}:$FILE"
    else
        echo "Warning: Node ID '$NODE_ID' not found in ALL_CLOUDLAB_HOSTS. Skipping."
    fi
done
