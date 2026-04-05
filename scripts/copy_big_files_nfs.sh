#!/bin/bash

# --- Configuration ---
FILES=(
    "base.1B.u8bin.crop_nb_100000000"
    "pipeann_100M_pq_compressed.bin"
    "pipeann_100M_pq_pivots.bin"
    "pipeann_100M_graph"
    "pipeann_100M_mem.index"
    "pipeann_100M_mem.index.data"
    "pipeann_100M_mem.index.tags"
)

SOURCE_DIR="/nfs/anngraphs/bigann/100M"
DEST_DIR="/mydata/local/anngraphs/bigann/100M"

# Bypasses the "Host key verification failed" prompts
SSH_OPT="-o StrictHostKeyChecking=no"

# The nodes in exact order of the daisy chain
NODES=("nfs" "node1" "node2" "node3" "node4")

# --- Execution ---

# 1. Phase 0: Local check and copy on nfs (node0)
echo ">>> Phase 0: Checking files locally on nfs (node0)..."
mkdir -p "$DEST_DIR"

for FILE in "${FILES[@]}"; do
    if [ ! -f "$DEST_DIR/$FILE" ]; then
        echo "Transferring $FILE to local nfs storage..."
        rsync -ah --sparse --inplace --progress "$SOURCE_DIR/$FILE" "$DEST_DIR/$FILE"
    else
        echo "✅ $FILE already exists on node0. Skipping rsync."
    fi
done

echo ""
echo ">>> Phase 0 Complete. Starting Daisy Chain..."

# 2. Phase 1: Daisy Chain to the rest of the nodes
for i in {1..4}; do
    PREV_NODE="${NODES[$((i-1))]}"
    CURR_NODE="${NODES[$i]}"
    
    echo "===================================================="
    echo ">>> Phase $i: $CURR_NODE pulling from $PREV_NODE"
    echo "===================================================="
    
    for FILE in "${FILES[@]}"; do
        # We send the 'if exists' logic directly to the remote node via SSH
        ssh $SSH_OPT namanh@$CURR_NODE "
            mkdir -p $DEST_DIR
            if [ ! -f $DEST_DIR/$FILE ]; then
                echo 'Moving $FILE to $CURR_NODE...'
                rsync -ah --sparse --inplace -e 'ssh $SSH_OPT' --progress namanh@$PREV_NODE:$DEST_DIR/$FILE $DEST_DIR/$FILE
            else
                echo '✅ $FILE already exists on $CURR_NODE. Skipping rsync.'
            fi
        "
    done
done

echo "🎉 Done! All files transferred to all 10 compute nodes."
