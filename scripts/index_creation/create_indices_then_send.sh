#!/bin/bash

# this file calls create_indices_v2 for all specified partitions then syncs them to Cloudlab


# sameple on fern: ./scripts/index_creation/create_indices_then_send.sh /ssd1/anndata/text2image1B/100M/global_partitions_5/pipeann_100M_partition 5 /ssd1/anndata/text2image1B/100M/base.1B.fbin.crop_nb_100000000 /ssd1/anndata/text2image1B/100M/100M_vamana_64_120_1.0 /ssd1/anndata/text2image1B/100M/ /ssd1/anndata/text2image1B/100M/pipeann_100M 64 128

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "${SCRIPT_DIR}/common_vars.sh"
source "${SCRIPT_DIR}/../cloudlab_addresses.sh"

DATA_TYPE=float
DATASET_NAME=text2image1B
DATASET_SIZE=100M
METRIC=mips
MODE="distributed"
ALPHA=1.0

PARTITION_IDS_PREFIX=$1
NUM_SERVERS=$2
BASE_FILE=$3
GRAPH_FILE=$4
DATA_FOLDER=$5
GLOBAL_INDEX_PREFIX=$6
SCATTER_GATHER_R=$7
SCATTER_GATHER_L=$8
MAX_NORM_FILE=${9:-""}

# Removed 'shift 1' as it isn't necessary unless you are iterating over remaining arguments

PARTITION_ASSIGNMENT_FILE="${PARTITION_IDS_PREFIX}_assignment.bin"

CLOUDLAB_DATA_FOLDER="/mydata/local/anngraphs/${DATASET_NAME}/${DATASET_SIZE}/"

NORMALIZED_BASE_FILE=$BASE_FILE
if [[ $METRIC == "mips" ]]; then
    NORMALIZED_BASE_FILE=${NORMALIZED_BASE_FILE}${NORMALIZED_SUFFIX}
fi
EXCLUDED_LIST=(0)

for ((i=0; i<$NUM_SERVERS; i++)); do
    # FIXED: Added the missing 'T' in PARTITION
    PARTITION_ID_FILE="${PARTITION_IDS_PREFIX}${i}_ids_uint32.bin"
    CLOUDLAB_HOST="${ALL_CLOUDLAB_HOSTS[$i]}"

# Check if the current server index is in the excluded list
    if [[ " ${EXCLUDED_LIST[*]} " =~ " $i " ]]; then
        echo "Skipping excluded server index: $i"
        continue
    fi    
    
    echo "Working on partition id file : $PARTITION_ID_FILE"
    echo "Working on $CLOUDLAB_HOST"

    "${SCRIPT_DIR}/create_graph_files.sh" $DATA_TYPE \
         $METRIC \
         $PARTITION_ID_FILE \
         $BASE_FILE \
         64 \
         128 \
	 $ALPHA \
         $DATA_FOLDER \
         $NUM_SERVERS \
         $MODE
    
    source "${SCRIPT_DIR}/create_indices_v2.sh" $DATASET_NAME \
         $DATASET_SIZE \
         $DATA_TYPE \
         $PARTITION_ID_FILE \
         $NORMALIZED_BASE_FILE \
         $GRAPH_FILE \
         $SCATTER_GATHER_R \
         $SCATTER_GATHER_L \
         $NUM_SERVERS \
         $MODE \
         $METRIC \
         $PARTITION_ASSIGNMENT_FILE \
         $DATA_FOLDER \
         $GLOBAL_INDEX_PREFIX \
	 $MAX_NORM_FILE
    
    # FIXED: Replaced '/' with ':' for remote rsync targets
    ssh $CLOUDLAB_HOST "mkdir -p $CLOUDLAB_DATA_FOLDER"
    rsync -av "$SCATTER_GATHER_OUTPUT" "${CLOUDLAB_HOST}:${CLOUDLAB_DATA_FOLDER}"
    rsync -av --no-links "$STATE_SEND_OUTPUT" "${CLOUDLAB_HOST}:${CLOUDLAB_DATA_FOLDER}"
    
    # send the pq data files + the partition assignment file to base location of the cloudlab host
    # also send the global mem index
    # this is just to make sure statesend has all the data it needs
    # need to not send any symlink files as we will symlink the data later
    
    # FIXED: Typo in COMRPESSED to COMPRESSED (Assuming it was misspelled)
    # FIXED: Replaced '/' with ':'
    rsync -v "$PQ_COMPRESSED_PATH" "${CLOUDLAB_HOST}:${CLOUDLAB_DATA_FOLDER}"
    rsync -v "$PQ_PIVOT_PATH" "${CLOUDLAB_HOST}:${CLOUDLAB_DATA_FOLDER}"
    rsync -v "${MEM_INDEX_PATH}"* "${CLOUDLAB_HOST}:${CLOUDLAB_DATA_FOLDER}"
    
    # need to delete all data after sending
    # FIXED: Removed brace expansion for safer variable deletion
    rm -rf "$SCATTER_GATHER_OUTPUT" "$STATE_SEND_OUTPUT" "$PARTITION_BASE_FILE_FOLDER" "$SCATTER_GATHER_GRAPH_FOLDER" "$STATE_SEND_GRAPH_FOLDER"
    
done
