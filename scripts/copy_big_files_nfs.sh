#!/bin/bash

# --- Configuration ---
# Add all your filenames here (space-separated)
FILES=(
    "vamana_64_128_1.2"
    "base.1B.u8bin"
    "indices.bin"
    "pipeann_1B_pq_compressed.bin"
    "pipeann_1B_pq_pivots.bin"
)

SOURCE_DIR="/nfs/anngraphs/bigann/1B"
DEST_DIR="/mydata/local/anngraphs/bigann/1B"
NODES=(0 1 2 3 4 5 6 7 8 9 10) # node0 to node10

# --- Execution ---

# 1. Step One: Node0 pulls ALL files from NFS
echo ">>> Phase 1: node0 pulling files from NFS..."
for FILE in "${FILES[@]}"; do
    echo "Transferring $FILE to node0..."
    ssh namanh@node0 "mkdir -p $DEST_DIR && rsync -ah --progress $SOURCE_DIR/$FILE $DEST_DIR/$FILE"
done

# 2. Step Two: Daisy Chain to the rest of the nodes
# node1 pulls from node0, node2 pulls from node1, etc.
for i in {1..10}; do
    PREV_NODE="node$((i-1))"
    CURR_NODE="node$i"
    
    echo "===================================================="
    echo ">>> Phase $((i+1)): $CURR_NODE pulling from $PREV_NODE"
    echo "===================================================="
    
    for FILE in "${FILES[@]}"; do
        echo "Moving $FILE..."
        # Pulling from the previous node's local NVMe to the current node's local NVMe
        ssh namanh@$CURR_NODE "mkdir -p $DEST_DIR && rsync -ah --progress namanh@$PREV_NODE:$DEST_DIR/$FILE $DEST_DIR/$FILE"
    done
done

echo "Done! All files transferred to all 11 nodes."
