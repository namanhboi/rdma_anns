#!/bin/bash
MYDATA_PREFIX=/mydata/local/anngraphs/bigann/100M/
NFS_PREFIX=/nfs/anngraphs/bigann/100M/
FOLDER_NAME=$1
ID=$2


MYDATA_FILES_PREFIX="${MYDATA_PREFIX}/${FOLDER_NAME}/pipeann_100M"
NFS_FILES_PREFIX="${NFS_PREFIX}/${FOLDER_NAME}/pipeann_100M"
cp "${NFS_FILES_PREFIX}${ID}_disk.index" "${MYDATA_FILES_PREFIX}_disk.index"
cp "${NFS_FILES_PREFIX}${ID}_ids_uint32.bin" "${MYDATA_FILES_PREFIX}_ids_uint32.bin"
cp "${NFS_FILES_PREFIX}_partition_assignment.bin" "${MYDATA_FILES_PREFIX}_partition_assignment.bin




