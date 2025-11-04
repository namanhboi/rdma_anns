#!/bin/bash
MYDATA_PREFIX=/mydata/local/anngraphs/bigann/100M/
NFS_PREFIX=/nfs/anngraphs/bigann/100M/
FOLDER_NAME=$1
ID=$2

# Determine suffix based on folder name
if [[ "$FOLDER_NAME" == *"partitions"* ]]; then
    SUFFIX="_partition${ID}"
elif [[ "$FOLDER_NAME" == *"clusters"* ]]; then
    SUFFIX="_cluster${ID}"
else
    echo "Error: Folder name must contain 'partitions' or 'clusters'"
    exit 1
fi

MYDATA_FILES_PREFIX="${MYDATA_PREFIX}/${FOLDER_NAME}/pipeann_100M"
NFS_FILES_PREFIX="${NFS_PREFIX}/${FOLDER_NAME}/pipeann_100M"

cp "${NFS_FILES_PREFIX}${SUFFIX}_disk.index" "${MYDATA_FILES_PREFIX}${SUFFIX}_disk.index"
cp "${NFS_FILES_PREFIX}${SUFFIX}_ids_uint32.bin" "${MYDATA_FILES_PREFIX}${SUFFIX}_ids_uint32.bin"
cp "${NFS_FILES_PREFIX}_partition_assignment.bin" "${MYDATA_FILES_PREFIX}_partition_assignment.bin"
