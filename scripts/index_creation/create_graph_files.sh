#!/usr/bin/bash

# this script should run on the server that makes the big graph for state send
# create graph files based on the partition id files and store them into a specific graph_files folder
# need to original non-normalized bin file (parlayann works with this)


# then use the other script to create the partition base flie + assemble graph + base file into partition indices + create pq data and mem index

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common_vars.sh


if [[ $# -ne 9 ]]; then
    echo "Usage: <data_type> <metric> <partition_id_file> <base_file> <R> <L> <data_folder> <num_partitions> <mode>"
    exit 1
fi


DATA_TYPE=$1
METRIC=$2
PARTITION_ID_FILE=$3
BASE_FILE=$4
R=$5
L=$6
# used to store all these different files. Store the big graph file at base of data_folder, partition graph folders in DATA_FOLDER/graph_files/global_partitions_{num_partitions}/... or DATA_FOLDER/graph_files/clusters_{num_partitions}/... 
DATA_FOLDER=$7
NUM_PARTITIONS=$8
MODE=$9




[[ "$MODE" != "local" && "$MODE" != "distributed" ]] && { echo "Error: mode must be local or distributed"; exit 1; }
[[ "$METRIC" != "l2" && "$METRIC" != "mips" ]] && { echo "Error: metric must be l2 or mips"; exit 1; }
if [[ $MODE == "local" ]]; then
    RAM_BUDGET=32
else 
    RAM_BUDGET=64
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "${DATA_FOLDER} doesn't exist"
    exit 1
fi

if [[ ! -f "$BASE_FILE" ]]; then
    echo "${BASE_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ID_FILE" ]]; then
    echo "${PARTITION_ID_FILE} doesn't exist"
    exit 1
fi


if [[ $METRIC == "mips" ]]; then
    if [[ $BASE_FILE == *"${NORMALIZED_SUFFIX}" ]]; then
	echo "for graph creation, base file provided must be non-normalized (aka not end with ${NORMALIZED_SUFFIX})"
	exit 1
    fi
fi

PARTITION_GRAPH_BASE_FOLDER=$DATA_FOLDER/graph_files/
if [[ ! -d $PARTITION_GRAPH_BASE_FOLDER ]]; then
    mkdir $PARTITION_GRAPH_BASE_FOLDER
fi

PARTITION_BASE_FILE_BASE_FOLDER=$DATA_FOLDER/base_files/
if [[ ! -d $PARTITION_BASE_FILE_BASE_FOLDER ]]; then
    mkdir $PARTITION_BASE_FILE_BASE_FOLDER
fi

# making folder to place the parition base file for scatter gather
if [[ ! -d "$PARTITION_BASE_FILE_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}/" ]]; then
    mkdir "$PARTITION_BASE_FILE_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}/"
fi




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
    echo "Could not extract data set size from $filename"
    exit 1
fi

PARTITION_STATE_SEND_GRAPH_FOLDER=$PARTITION_GRAPH_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}
PARTITION_SCATTER_GATHER_GRAPH_FOLDER=$PARTITION_GRAPH_BASE_FOLDER/clusters_${NUM_PARTITIONS}

if [[ ! -d $PARTITION_STATE_SEND_GRAPH_FOLDER ]]; then
    mkdir $PARTITION_STATE_SEND_GRAPH_FOLDER
fi

if [[ ! -d $PARTITION_SCATTER_GATHER_GRAPH_FOLDER ]]; then
    mkdir $PARTITION_SCATTER_GATHER_GRAPH_FOLDER
fi


if [[ "$MODE" == "local" ]]; then
    ALPHA=1.0
else
    ALPHA=1.2
fi
   

GLOBAL_PARLAYANN_GRAPH=$DATA_FOLDER/vamana_${R}_${L}_${ALPHA}
if [[ ! -f $GLOBAL_PARLAYANN_GRAPH ]]; then
    echo "Creating the global grpah file here: $GLOBAL_PARLAYANN_GRAPH" 
    $WORKDIR/extern/ParlayANN/algorithms/vamana/neighbors -R $R -L $L -alpha $ALPHA -two_pass 0 -graph_outfile $GLOBAL_PARLAYANN_GRAPH -data_type $DATA_TYPE -dist_func $METRIC -base_path $BASE_FILE
fi


# now we need to partition the graph for statesend
PARTITION_STATE_SEND_GRAPH_FILE=$PARTITION_STATE_SEND_GRAPH_FOLDER/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph
if [[ ! -f "${PARTITION_STATE_SEND_GRAPH_FILE}" ]]; then
    "${WORKDIR}/build/src/state_send/create_partition_graph_file" \
	"${GLOBAL_PARLAYANN_GRAPH}" \
	"${PARTITION_ID_FILE}" \
	"${PARTITION_STATE_SEND_GRAPH_FILE}"

    # rm "${PARTITION_STATE_SEND_GRAPH_FILE}_parlayann"
fi

echo "Done with creating graph files for statesend"


# now we need to build the graph for scatter gather
# first need to make the partition base file, and then delete that shit later
PARTITION_SCATTER_GATHER_BASE_FILE=$PARTITION_BASE_FILE_BASE_FOLDER/global_partitions_${NUM_PARTITIONS}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}.bin

if [[ ! -f $PARTITION_SCATTER_GATHER_BASE_FILE ]]; then
    "${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
	"${DATA_TYPE}" \
	"${BASE_FILE}" \
	"${PARTITION_ID_FILE}" \
	"${PARTITION_SCATTER_GATHER_BASE_FILE}"
fi

PARTITION_SCATTER_GATHER_GRAPH_FILE=$PARTITION_GRAPH_BASE_FOLDER/clusters_${NUM_PARTITIONS}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}_graph
if [[ ! -f $PARTITION_SCATTER_GATHER_GRAPH_FILE ]]; then
    $WORKDIR/extern/ParlayANN/algorithms/vamana/neighbors -R $R -L $L -alpha $ALPHA -two_pass 0 -graph_outfile "${PARTITION_SCATTER_GATHER_GRAPH_FILE}_parlayann" -data_type $DATA_TYPE -dist_func $METRIC -base_path $PARTITION_SCATTER_GATHER_BASE_FILE

    "${WORKDIR}/build/src/state_send/convert_parlayann_graph_file" \
	"${PARTITION_SCATTER_GATHER_GRAPH_FILE}_parlayann" \
	"${PARTITION_SCATTER_GATHER_GRAPH_FILE}"
fi


echo "Done with creating graph files for scatter gather"

# rm "$PARTITION_SCATTER_GATHER_BASE_FILE"

