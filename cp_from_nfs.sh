#!/bin/bash
MYDATA_PREFIX=/mydata/local/anngraphs/bigann/100M/
NFS_PREFIX=/nfs/anngraphs/bigann/100M/
FOLDER_NAME=$1
ID=$2

MYDATA_FILES_PREFIX="${MYDATA_PREFIX}/${FOLDER_NAME}/pipeann_100M"
NFS_FILES_PREFIX="${NFS_PREFIX}/${FOLDER_NAME}/pipeann_100M"

# Determine suffix and copy logic based on folder name
if [[ "$FOLDER_NAME" == *"partitions"* ]]; then
    SUFFIX="_partition${ID}"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_disk.index" "${MYDATA_FILES_PREFIX}${SUFFIX}_disk.index"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_ids_uint32.bin" "${MYDATA_FILES_PREFIX}${SUFFIX}_ids_uint32.bin"
    cp "${NFS_FILES_PREFIX}_partition_assignment.bin" "${MYDATA_FILES_PREFIX}_partition_assignment.bin"
    
elif [[ "$FOLDER_NAME" == *"clusters"* ]]; then
    SUFFIX="_cluster${ID}"
    # Copy all files with the cluster suffix except those ending with _graph
    for file in "${NFS_FILES_PREFIX}${SUFFIX}"*; do
        if [[ ! "$file" == *"_graph" ]]; then
            filename=$(basename "$file")
            cp "$file" "${MYDATA_FILES_PREFIX}${SUFFIX}${filename#*${SUFFIX}}"
        fi
    done
    
else
    echo "Error: Folder name must contain 'partitions' or 'clusters'"
    exit 1
fi
