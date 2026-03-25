#!/bin/bash

# Assuming that the data is in namanh@nfs:/mydata/local/anngraphs/{dataset_name}/{scale}
# This file will create the indices for both the scatter gather and state send
# approach from the partition, graph (parlayann), and datafile and put them in the specified folders

set -euo pipefail

SOURCED=0
(return 0 2>/dev/null) && SOURCED=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "${SCRIPT_DIR}/common_vars.sh"

if [[ $# -ne 15 && $# -ne 16 ]]; then
    echo "Usage: ${BASH_SOURCE[0]} <dataset_name> <dataset_size> <data_type> <partition_file> <base_file> <graph_file> <scatter_gather_r> <scatter_gather_l> <num_servers> <mode> <metric> <partition_assignment_file> <data_folder> <global_index_prefix> <is_random_partitioning> <max_norm_file:optional>"
    exit 1
fi

DATASET_NAME=$1
DATASET_SIZE=$2
DATA_TYPE=$3
PARTITION_ID_FILE=$4
BASE_FILE=$5
GRAPH_FILE=$6
# SCATTER_GATHER_OUTPUT=$7
SCATTER_GATHER_R=$7
SCATTER_GATHER_L=$8
# STATE_SEND_OUTPUT=${10}
NUM_SERVERS=$9
MODE=${10}
METRIC=${11}
PARTITION_ASSIGNMENT_FILE=${12}
DATA_FOLDER=${13}
GLOBAL_INDEX_PREFIX=${14}
IS_RANDOM_PARTITION=${15}
MAX_NORM_FILE=${16:-""}

NUM_THREADS=56
MEM_INDEX_SAMPLING_RATE=0.01
MEM_INDEX_R=32
MEM_INDEX_L=64
MEM_INDEX_ALPHA=1.2
SCATTER_GATHER_ALPHA=1.2
NUM_PQ_CHUNKS=32

[[ "$DATASET_NAME" != "bigann" && "$DATASET_NAME" != "deep1b" && "$DATASET_NAME" != "MSSPACEV1B" && "$DATASET_NAME" != "text2image1B" ]] && { echo "Error: dataset_name must be 'bigann, deep1b, MSSPACEV1B, text2image1B'"; exit 1; }
[[ "$MODE" != "local" && "$MODE" != "distributed" ]] && { echo "Error: mode must be local or distributed"; exit 1; }
[[ "$METRIC" != "l2" && "$METRIC" != "mips" ]] && { echo "Error: metric must be l2 or mips"; exit 1; }

if [[ "$METRIC" == "mips" && "$MAX_NORM_FILE" == "" ]]; then
    echo "Error: max norm file can't be empty if using mips"
    exit 1
fi

if [[ "$MODE" == "local" ]]; then
    RAM_BUDGET=32
else 
    RAM_BUDGET=64
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "Error: ${DATA_FOLDER} doesn't exist"
    exit 1
fi

if [[ ! -f "$GRAPH_FILE" ]]; then
    echo "Error: ${GRAPH_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$BASE_FILE" ]]; then
    echo "Error: ${BASE_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ASSIGNMENT_FILE" ]]; then
    echo "Error: ${PARTITION_ASSIGNMENT_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ID_FILE" ]]; then
    echo "Error: ${PARTITION_ID_FILE} doesn't exist"
    exit 1
fi

if [[ "$METRIC" == "mips" ]]; then
    if [[ "$BASE_FILE" != *"${NORMALIZED_SUFFIX}" ]]; then
        echo "Error: mips requires the base file ($BASE_FILE) to be normalized (aka end with ${NORMALIZED_SUFFIX})"
        exit 1
    fi
fi

# Dynamically route ALL inputs and outputs based on the partitioning strategy
if [[ "$IS_RANDOM_PARTITION" == "true" ]]; then 
    SCATTER_GATHER_OUTPUT="$DATA_FOLDER/clusters_random_${NUM_SERVERS}"
    STATE_SEND_OUTPUT="$DATA_FOLDER/global_random_partitions_${NUM_SERVERS}"
    PARTITION_BASE_FILE_FOLDER="${DATA_FOLDER}/base_files/global_random_partitions_${NUM_SERVERS}"
    SCATTER_GATHER_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/clusters_random_${NUM_SERVERS}"
    STATE_SEND_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/global_random_partitions_${NUM_SERVERS}"
else 
    SCATTER_GATHER_OUTPUT="$DATA_FOLDER/clusters_${NUM_SERVERS}"
    STATE_SEND_OUTPUT="$DATA_FOLDER/global_partitions_${NUM_SERVERS}"
    PARTITION_BASE_FILE_FOLDER="${DATA_FOLDER}/base_files/global_partitions_${NUM_SERVERS}"
    SCATTER_GATHER_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/clusters_${NUM_SERVERS}"
    STATE_SEND_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/global_partitions_${NUM_SERVERS}"
fi

mkdir -p "$SCATTER_GATHER_OUTPUT"
mkdir -p "$STATE_SEND_OUTPUT"
mkdir -p "$PARTITION_BASE_FILE_FOLDER"

if [[ ! -d "$SCATTER_GATHER_GRAPH_FOLDER" ]]; then
    echo "Error: $SCATTER_GATHER_GRAPH_FOLDER doesn't exist, need to run create_graph_files.sh first"
    exit 1
fi

if [[ ! -d "$STATE_SEND_GRAPH_FOLDER" ]]; then
    echo "Error: $STATE_SEND_GRAPH_FOLDER doesn't exist, need to run create_graph_files.sh first"
    exit 1
fi

filename=$(basename "$PARTITION_ID_FILE" .bin)
if [[ "$filename" =~ partition([0-9]+) ]]; then
    PARTITION_NUM="${BASH_REMATCH[1]}"
else
    echo "Error: Could not extract partition number from $filename"
    exit 1
fi

STATE_SEND_INDEX_PREFIX="${STATE_SEND_OUTPUT}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}"
SCATTER_GATHER_INDEX_PREFIX="${SCATTER_GATHER_OUTPUT}/pipeann_${DATASET_SIZE}_cluster${PARTITION_NUM}"

# Slicing the big base file into the partition base file
PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_FOLDER}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}"
if [[ "$METRIC" == "mips" ]]; then
    PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_PATH}.bin${NORMALIZED_SUFFIX}"
else
    PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_PATH}.bin"
fi

echo "Partition base file path is $PARTITION_BASE_FILE_PATH"
if [[ ! -f "${PARTITION_BASE_FILE_PATH}" ]]; then 
    "${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
        "${DATA_TYPE}" \
        "${BASE_FILE}" \
        "${PARTITION_ID_FILE}" \
        "${PARTITION_BASE_FILE_PATH}"
fi

PARTITION_SCATTER_GATHER_GRAPH_FILE="$SCATTER_GATHER_GRAPH_FOLDER/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph"
if [[ ! -f "$PARTITION_SCATTER_GATHER_GRAPH_FILE" ]]; then
    echo "Error: $PARTITION_SCATTER_GATHER_GRAPH_FILE doesn't exist, need to run create_graph_files.sh"
    exit 1
fi

PARTITION_STATE_SEND_GRAPH_FILE="${STATE_SEND_GRAPH_FOLDER}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph"
if [[ ! -f "$PARTITION_STATE_SEND_GRAPH_FILE" ]]; then
    echo "Error: $PARTITION_STATE_SEND_GRAPH_FILE doesn't exist, need to run create_graph_files.sh"
    exit 1
fi

echo "Begin creating scatter gather index..."
"${SCRIPT_DIR}/create_scatter_gather_index.sh" \
    "$DATA_TYPE" \
    "$METRIC" \
    "$SCATTER_GATHER_R" \
    "$SCATTER_GATHER_L" \
    "$NUM_PQ_CHUNKS" \
    "$RAM_BUDGET" \
    "$NUM_THREADS" \
    "$SCATTER_GATHER_INDEX_PREFIX" \
    "$MEM_INDEX_SAMPLING_RATE" \
    "$PARTITION_ID_FILE" \
    "$PARTITION_BASE_FILE_PATH" \
    "$PARTITION_SCATTER_GATHER_GRAPH_FILE" \
    "$MAX_NORM_FILE"

# Now create STATE_SEND index infrastructure
MEM_INDEX_PATH="${GLOBAL_INDEX_PREFIX}_mem.index"
if [[ ! -f "${MEM_INDEX_PATH}" ]]; then
    echo "Mem index at ${MEM_INDEX_PATH} doesn't exist."
    echo "Creating global memory index..."
    SLICE_PREFIX="${GLOBAL_INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
    
    "${WORKDIR}/build/src/state_send/gen_random_slice" \
        "${DATA_TYPE}" \
        "${BASE_FILE}" \
        "${SLICE_PREFIX}" \
        "${MEM_INDEX_SAMPLING_RATE}"

    SLICE_TAG="${SLICE_PREFIX}_ids.bin"   
    
    if [[ "$METRIC" == "mips" ]]; then
        SLICE_DATA="${SLICE_PREFIX}${NORMALIZED_SUFFIX}"
    else
        SLICE_DATA="${SLICE_PREFIX}_data.bin"
    fi    

    "${WORKDIR}/build/src/state_send/build_memory_index" \
        "${DATA_TYPE}" \
        "${SLICE_DATA}" \
        "${SLICE_TAG}" \
        "${MEM_INDEX_R}" \
        "${MEM_INDEX_L}" \
        "${MEM_INDEX_ALPHA}" \
        "${MEM_INDEX_PATH}" \
        "${NUM_THREADS}" \
        "${METRIC}"
fi

# Check if global PQ is created, if not create it
PQ_COMPRESSED_PATH="${GLOBAL_INDEX_PREFIX}_pq_compressed.bin"
PQ_PIVOT_PATH="${GLOBAL_INDEX_PREFIX}_pq_pivots.bin"
if [[ (! -f "${PQ_COMPRESSED_PATH}") || (! -f "${PQ_PIVOT_PATH}") ]]; then
    echo "Creating global PQ data..."
    "${WORKDIR}/build/src/state_send/create_pq_data" \
        "$DATA_TYPE" \
        "$BASE_FILE" \
        "$GLOBAL_INDEX_PREFIX" \
        "$METRIC" \
        "$NUM_PQ_CHUNKS" 
fi

echo "Begin creating state send index..."
"${SCRIPT_DIR}/create_state_send_index.sh" \
    "$DATA_TYPE" \
    "$METRIC" \
    "$STATE_SEND_INDEX_PREFIX" \
    "$PARTITION_ID_FILE" \
    "$PARTITION_BASE_FILE_PATH" \
    "$PARTITION_STATE_SEND_GRAPH_FILE" \
    "$PARTITION_ASSIGNMENT_FILE" \
    "$GLOBAL_INDEX_PREFIX" \
    "$MAX_NORM_FILE"

if [[ $SOURCED -eq 1 ]]; then
    export SCATTER_GATHER_OUTPUT STATE_SEND_OUTPUT PARTITION_BASE_FILE_FOLDER SCATTER_GATHER_GRAPH_FOLDER STATE_SEND_GRAPH_FOLDER
    export PQ_COMPRESSED_PATH PQ_PIVOT_PATH MEM_INDEX_PATH
fi
