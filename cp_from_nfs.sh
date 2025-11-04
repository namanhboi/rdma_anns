#!/bin/bash
MYDATA_PREFIX=/mydata/local/anngraphs/bigann/100M/
NFS_PREFIX=/nfs/anngraphs/bigann/100M/
FOLDER_NAME=$1
ID=$2

mkdir -p "${MYDATA_PREFIX}/${FOLDER_NAME}"

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
    # Copy specific cluster files
    cp "${NFS_FILES_PREFIX}${SUFFIX}_disk.index" "${MYDATA_FILES_PREFIX}${SUFFIX}_disk.index"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_disk.index.tags" "${MYDATA_FILES_PREFIX}${SUFFIX}_disk.index.tags"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_mem.index" "${MYDATA_FILES_PREFIX}${SUFFIX}_mem.index"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_mem.index.data" "${MYDATA_FILES_PREFIX}${SUFFIX}_mem.index.data"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_mem.index.tags" "${MYDATA_FILES_PREFIX}${SUFFIX}_mem.index.tags"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_pq_compressed.bin" "${MYDATA_FILES_PREFIX}${SUFFIX}_pq_compressed.bin"
    cp "${NFS_FILES_PREFIX}${SUFFIX}_pq_pivots.bin" "${MYDATA_FILES_PREFIX}${SUFFIX}_pq_pivots.bin"
    cp "${NFS_FILES_PREFIX}${SUFFIX}.tags" "${MYDATA_FILES_PREFIX}${SUFFIX}.tags"
    
else
    echo "Error: Folder name must contain 'partitions' or 'clusters'"
    exit 1
fi
