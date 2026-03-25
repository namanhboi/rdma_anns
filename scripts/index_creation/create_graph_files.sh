#!/usr/bin/bash

# This script should run on the server that makes the big graph for state send
# Create graph files based on the partition id files and store them into a specific graph_files folder
# Needs the original non-normalized bin file (parlayann works with this)

# Then use the other script to create the partition base file + assemble graph + base file into partition indices + create pq data and mem index

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/common_vars.sh"

# FIX: Changed to 11 to account for IS_RANDOM_PARTITION
if [[ $# -ne 11 ]]; then
    echo "Usage: <data_type> <metric> <partition_id_file> <base_file> <R> <L> <ALPHA> <data_folder> <num_partitions> <mode> <is_random_partition>"
    exit 1
fi

DATA_TYPE=$1
METRIC=$2
PARTITION_ID_FILE=$3
BASE_FILE=$4
R=$5
L=$6
ALPHA=$7
# Used to store all these different files. Store the big graph file at base of data_folder...
DATA_FOLDER=$8
NUM_PARTITIONS=$9
MODE=${10}
IS_RANDOM_PARTITION=${11}

[[ "$MODE" != "local" && "$MODE" != "distributed" ]] && { echo "Error: mode must be local or distributed"; exit 1; }
[[ "$METRIC" != "l2" && "$METRIC" != "mips" ]] && { echo "Error: metric must be l2 or mips"; exit 1; }

if [[ $MODE == "local" ]]; then
    RAM_BUDGET=32
else 
    RAM_BUDGET=64
fi

if [[ $METRIC == "l2" ]]; then
    METRIC="Euclidian" # Check if your ParlayANN expects "Euclidian" or "Euclidean"
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "Error: ${DATA_FOLDER} doesn't exist"
    exit 1
fi

if [[ ! -f "$BASE_FILE" ]]; then
    echo "Error: ${BASE_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ID_FILE" ]]; then
    echo "Error: ${PARTITION_ID_FILE} doesn't exist"
    exit 1
fi

if [[ $METRIC == "mips" ]]; then
    if [[ "$BASE_FILE" == *"${NORMALIZED_SUFFIX}" ]]; then
        echo "Error: for graph creation, base file provided must be non-normalized (aka not end with ${NORMALIZED_SUFFIX}): $BASE_FILE"
        exit 1
    fi
fi

# FIX: Using mkdir -p and quoting variables
PARTITION_GRAPH_BASE_FOLDER="$DATA_FOLDER/graph_files/"
mkdir -p "$PARTITION_GRAPH_BASE_FOLDER"

PARTITION_BASE_FILE_BASE_FOLDER="$DATA_FOLDER/base_files/"
mkdir -p "$PARTITION_BASE_FILE_BASE_FOLDER"

# Making folder to place the partition base file for scatter gather
if [[ "$IS_RANDOM_PARTITION" == "true" ]]; then
    PARTITION_SCATTER_GATHER_BASE_FOLDER="$PARTITION_BASE_FILE_BASE_FOLDER/global_random_partitions_${NUM_PARTITIONS}/"
else
    PARTITION_SCATTER_GATHER_BASE_FOLDER="$PARTITION_BASE_FILE_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}/"
fi

mkdir -p "$PARTITION_SCATTER_GATHER_BASE_FOLDER"

filename=$(basename "$PARTITION_ID_FILE" .bin)
if [[ "$filename" =~ partition([0-9]+) ]]; then
    PARTITION_NUM="${BASH_REMATCH[1]}"
    echo "Processing partition: (number: $PARTITION_NUM)"
else
    echo "Error: Could not extract partition number from $filename"
    exit 1
fi

if [[ "$filename" =~ pipeann_([^_]+)_ ]]; then
    DATASET_SIZE="${BASH_REMATCH[1]}"
    echo "Dataset size is $DATASET_SIZE"
else
    echo "Error: Could not extract data set size from $filename"
    exit 1
fi

# FIX: Logic was inverted here. Now "true" correctly routes to random partitions.
if [[ "$IS_RANDOM_PARTITION" == "true" ]]; then
    PARTITION_STATE_SEND_GRAPH_FOLDER="$PARTITION_GRAPH_BASE_FOLDER/global_random_partitions_${NUM_PARTITIONS}"
    PARTITION_SCATTER_GATHER_GRAPH_FOLDER="$PARTITION_GRAPH_BASE_FOLDER/clusters_random_${NUM_PARTITIONS}"
else
    PARTITION_STATE_SEND_GRAPH_FOLDER="$PARTITION_GRAPH_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}"
    PARTITION_SCATTER_GATHER_GRAPH_FOLDER="$PARTITION_GRAPH_BASE_FOLDER/clusters_${NUM_PARTITIONS}"
fi

mkdir -p "$PARTITION_STATE_SEND_GRAPH_FOLDER"
mkdir -p "$PARTITION_SCATTER_GATHER_GRAPH_FOLDER"

GLOBAL_PARLAYANN_GRAPH="$DATA_FOLDER/vamana_${R}_${L}_${ALPHA}"
if [[ ! -f "$GLOBAL_PARLAYANN_GRAPH" ]]; then
    GLOBAL_GRAPH="$DATA_FOLDER/pipeann_${DATASET_SIZE}_graph"
    if [[ ! -f "$GLOBAL_GRAPH" ]]; then
        echo "Creating the global graph file here: $GLOBAL_PARLAYANN_GRAPH" 
        "$WORKDIR/extern/ParlayANN/algorithms/vamana/neighbors" -R "$R" -L "$L" -alpha "$ALPHA" -two_pass 0 -graph_outfile "$GLOBAL_PARLAYANN_GRAPH" -data_type "$DATA_TYPE" -dist_func "$METRIC" -base_path "$BASE_FILE"  
    else
        GLOBAL_PARLAYANN_GRAPH="$GLOBAL_GRAPH"
    fi
fi

# Now we need to partition the graph for statesend
PARTITION_STATE_SEND_GRAPH_FILE="$PARTITION_STATE_SEND_GRAPH_FOLDER/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph"
if [[ ! -f "${PARTITION_STATE_SEND_GRAPH_FILE}" ]]; then
    "${WORKDIR}/build/src/state_send/create_partition_graph_file" \
        "${GLOBAL_PARLAYANN_GRAPH}" \
        "${PARTITION_ID_FILE}" \
        "${PARTITION_STATE_SEND_GRAPH_FILE}"
fi

echo "Done with creating graph files for statesend"

# Now we need to build the graph for scatter gather
# First need to make the partition base file, and then delete that later
PARTITION_SCATTER_GATHER_BASE_FILE="$PARTITION_SCATTER_GATHER_BASE_FOLDER/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}.bin"

if [[ ! -f "$PARTITION_SCATTER_GATHER_BASE_FILE" ]]; then
    "${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
        "${DATA_TYPE}" \
        "${BASE_FILE}" \
        "${PARTITION_ID_FILE}" \
        "${PARTITION_SCATTER_GATHER_BASE_FILE}"
fi

PARTITION_SCATTER_GATHER_GRAPH_FILE="$PARTITION_SCATTER_GATHER_GRAPH_FOLDER/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph"
if [[ ! -f "$PARTITION_SCATTER_GATHER_GRAPH_FILE" ]]; then
    "$WORKDIR/extern/ParlayANN/algorithms/vamana/neighbors" -R "$R" -L "$L" -alpha "$ALPHA" -two_pass 0 -graph_outfile "${PARTITION_SCATTER_GATHER_GRAPH_FILE}_parlayann" -data_type "$DATA_TYPE" -dist_func "$METRIC" -base_path "$PARTITION_SCATTER_GATHER_BASE_FILE"

    "${WORKDIR}/build/src/state_send/convert_parlayann_graph_file" \
        "${PARTITION_SCATTER_GATHER_GRAPH_FILE}_parlayann" \
        "${PARTITION_SCATTER_GATHER_GRAPH_FILE}"
fi

echo "Done with creating graph files for scatter gather"
