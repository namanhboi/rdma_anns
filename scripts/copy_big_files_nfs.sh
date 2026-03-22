#!/bin/bash

FILES=("vamana_64_128_1.2") # Add other files separated by spaces
SOURCE_DIR="/nfs/anngraphs/bigann/1B"
DEST_DIR="/mydata/local/anngraphs/bigann/1B"

# --- Phase 0: node0 pulls from NFS using cp ---
echo ">>> Phase 0: node0 pulling files from NFS..."
for FILE in "${FILES[@]}"; do
    echo "Copying $FILE to node0..."
    
    # We remove the old file, create the directory, and use cp with the sparse flag
    ssh namanh@node0 "rm -f $DEST_DIR/$FILE && mkdir -p $DEST_DIR && cp -v --sparse=always $SOURCE_DIR/$FILE $DEST_DIR/$FILE"
done

# --- Phase 1: Daisy chain to node1 -> node10 ---
for i in {1..10}; do
    PREV="node$((i-1))"
    CURR="node$i"
    
    echo "===================================================="
    echo ">>> Phase $i: $CURR pulling from $PREV"
    echo "===================================================="
    
    for FILE in "${FILES[@]}"; do
        echo "Moving $FILE..."
        
        # We keep rsync here for the node-to-node transfer to retain the progress bar and checksum safety
        ssh namanh@$CURR "rm -f $DEST_DIR/$FILE && mkdir -p $DEST_DIR && rsync -ah --inplace --progress namanh@$PREV:$DEST_DIR/$FILE $DEST_DIR/$FILE"
    done
done

echo "Done! All files transferred to all 11 nodes."
