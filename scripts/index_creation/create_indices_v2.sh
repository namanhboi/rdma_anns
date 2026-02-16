#!/bin/bash

# assuming that the data is in namanh@nfs:/mydata/local/anngraphs/{dataset_name}/{scale}
# this file will create the indices for both the scatter gather and state send
# approach from the partition, graph (parlayann), and datafile and put them in the specified folders

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ${SCRIPT_DIR}/common_vars.sh

if [ $# -ne 15 -a $# -ne 16 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <dataset_name> <dataset_size> <data_type> <partition_file> <base_file> <graph_file> <scatter_gather_output> <scatter_gather_r> <scatter_gather_l> <state_send_output> <mode> <metric> <partition_assignment_file> <max_norm_file>"
    echo "  dataset_name: bigann"
    echo "  dataset_size: 10M or 100M or 1B"
    echo "  data_type: uint8 or int8 or float"
    echo "  partition_id_file: /mydata/local/anngraphs/bigann/1B/global_partitions_5/pipeann_1B_partition0_ids_uint32.bin"
    echo "  base_file: /mydata/local/anngraphs/bigann/1B/base.1B.u8bin"
    echo "  graph_file: /mydata/local/anngraphs/bigann/1B/vamana_64_128_1.2"
    echo "  scatter_gather_output: /mydata/local/anngraphs/bigann/1B/clusters_5/"
    echo "  scatter_gather_r: 64"
    echo "  scatter_gather_l: 128"
    echo "  state_send_output: /mydata/local/anngraphs/bigann/1B/global_partitions_5/"
    echo "  mode: local or distributed"
    echo "  metric: l2, mips"
    echo "  partition_assignment_file:  /home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M_partition_assignment.bin"
    echo "  data_folder: folder to store the data and graph file created, for local it shuold be $HOME/big-ann-benchmarks/data/<DATASET_NAME>/<DATASET_SIZE>/ , on cloudlab /mydata/local/anngraphs/<DATASET_NAME>/<DATASET_SIZE>"
    echo "  global_index_prefix: used to get the mem index and pq data"
    echo "  MAX_NORM_FILE: used for mips, can leave blank "
    exit 1
fi


DATASET_NAME=$1
DATASET_SIZE=$2
DATA_TYPE=$3
PARTITION_ID_FILE=$4
BASE_FILE=$5
GRAPH_FILE=$6
SCATTER_GATHER_OUTPUT=$7
SCATTER_GATHER_R=$8
SCATTER_GATHER_L=$9
STATE_SEND_OUTPUT=${10}
MODE=${11}
METRIC=${12}
PARTITION_ASSIGNMENT_FILE=${13}
DATA_FOLDER=${14}
GLOBAL_INDEX_PREFIX=${15}
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

if [[ "$METRIC" == "mips" && $MAX_NORM_FILE == "" ]]; then
    echo "max norm file can't be empty if using mips"
    exit 1
fi


if [[ $MODE == "local" ]]; then
    RAM_BUDGET=32
else 
    RAM_BUDGET=64
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "${DATA_FOLDER} doesn't exist"
    exit 1
fi

if [[ ! -f "$GRAPH_FILE" ]]; then
    echo "${GRAPH_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$BASE_FILE" ]]; then
    echo "${BASE_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ASSIGNMENT_FILE" ]]; then
    echo "${PARTITION_ASSIGNMENT_FILE} doesn't exist"
    exit 1
fi

if [[ ! -f "$PARTITION_ID_FILE" ]]; then
    echo "${PARTITION_ID_FILE} doesn't exist"
    exit 1
fi

if [[ ! -d "$SCATTER_GATHER_OUTPUT" ]]; then
    mkdir "$SCATTER_GATHER_OUTPUT"
fi

if [[ ! -d "$STATE_SEND_OUTPUT" ]]; then
    mkdir "$STATE_SEND_OUTPUT"
fi


if [[ $METRIC == "mips" ]]; then
    if [[ $BASE_FILE != *"${NORMALIZED_SUFFIX}" ]]; then
	echo "mips requires the base file ($BASE_FILE) to be normalized (aka end with ${NORMALIZED_SUFFIX})"
	exit 1
    fi
fi




filename=$(basename "$PARTITION_ID_FILE" .bin)
if [[ "$filename" =~ partition([0-9]+) ]]; then
    PARTITION_NUM="${BASH_REMATCH[1]}"
    PARTITION_ID="partition${PARTITION_NUM}"
    echo "Processing partition: $PARTITION_ID (number: $PARTITION_NUM)"
else
    echo "Error: Could not extract partition number from $filename"
    exit 1
fi


BASE_AND_GRAPH_FILE_DIRNAME=$(basename "$STATE_SEND_OUTPUT")
STATE_SEND_INDEX_PREFIX="${STATE_SEND_OUTPUT}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}"
SCATTER_GATHER_INDEX_PREFIX="${SCATTER_GATHER_OUTPUT}/pipeann_${DATASET_SIZE}_cluster${PARTITION_NUM}"

# making directory to store all the bin files
PARTITION_BASE_FILE_FOLDER="${DATA_FOLDER}/base_files/${BASE_AND_GRAPH_FILE_DIRNAME}"
mkdir -p "${PARTITION_BASE_FILE_FOLDER}"

# here we are slicing the big base file into the partition base file based on the partition ids file
# We need to check if the big base file we provided is normalized or not
PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_FOLDER}/pipeann_${DATASET_SIZE}_partition${PARTITION_NUM}"
if [[ "$METRIC" == "mips" ]]; then
    PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_PATH}.bin${NORMALIZED_SUFFIX}"
else
    PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_PATH}.bin"
fi


echo "partition base file path is $PARTITION_BASE_FILE_PATH"
if [[ ! -f "${PARTITION_BASE_FILE_PATH}" ]]; then 
"${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
    "${DATA_TYPE}" \
    "${BASE_FILE}" \
    "${PARTITION_ID_FILE}" \
    "${PARTITION_BASE_FILE_PATH}"
fi

# NOW, we begin creating the ScatterGather index
${SCRIPT_DIR}/create_scatter_gather_index.sh \
	     $DATA_TYPE \
	     $METRIC \
	     $SCATTER_GATHER_R \
	     $SCATTER_GATHER_L \
	     $NUM_PQ_CHUNKS \
	     $RAM_BUDGET \
	     $NUM_THREADS \
	     $SCATTER_GATHER_INDEX_PREFIX \
	     $MEM_INDEX_SAMPLING_RATE \
	     $PARTITION_ID_FILE \
	     $PARTITION_BASE_FILE_PATH \
	     $MAX_NORM_FILE


# Now create STATE_SEND indice
# first need to create the partition graph file
PARTITION_STATE_SEND_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/${BASE_AND_GRAPH_FILE_DIRNAME}"
mkdir -p "${PARTITION_STATE_SEND_GRAPH_FOLDER}"
PARTITION_STATE_SEND_GRAPH_FILE="${PARTITION_STATE_SEND_GRAPH_FOLDER}/pipeann_${DATASET_SIZE}_${PARTITION_ID}_graph"

if [[ ! -f "${PARTITION_STATE_SEND_GRAPH_FILE}" ]]; then
    "${WORKDIR}/build/src/state_send/create_partition_graph_file" \
	"${GRAPH_FILE}" \
	"${PARTITION_ID_FILE}" \
	"${PARTITION_STATE_SEND_GRAPH_FILE}"
fi


# check if global mem index is created, if not create it
MEM_INDEX_PATH="${GLOBAL_INDEX_PREFIX}_mem.index"
if [[ ! -f "${MEM_INDEX_PATH}" ]]; then
    echo "mem index at ${MEM_INDEX_PATH} doesnt exist"
    echo "Creating global memory index..."
    SLICE_PREFIX="${GLOBAL_INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
    "${WORKDIR}/build/src/state_send/gen_random_slice" \
	"${DATA_TYPE}" \
	"${BASE_FILE}" \
	"${SLICE_PREFIX}" \
	"${MEM_INDEX_SAMPLING_RATE}"

    SLICE_TAG="${SLICE_PREFIX}_ids.bin"   
    
    if [[ $METRIC == "mips" ]]; then
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



# check if global pq is created, if not create it
PQ_COMPRESSED_PATH="${GLOBAL_INDEX_PREFIX}_pq_compressed.bin"
PQ_PIVOT_PATH="${GLOBAL_INDEX_PREFIX}_pq_pivots.bin"
if [[ (! -f "${PQ_COMPRESSED_PATH}") || (! -f "${PQ_PIVOT_PATH}") ]]; then
    # create global pq data
    "$WORKDIR/build/src/state_send/create_pq_data" \
	$DATA_TYPE \
	$BASE_FILE \
	$GLOBAL_INDEX_PREFIX \
	$METRIC \
	$NUM_PQ_CHUNKS 
fi



$SCRIPT_DIR/create_state_send_index.sh \
    $DATA_TYPE \
    $METRIC \
    $STATE_SEND_INDEX_PREFIX \
    $PARTITION_ID_FILE \
    $PARTITION_BASE_FILE_PATH \
    $PARTITION_STATE_SEND_GRAPH_FILE \
    $PARTITION_ASSIGNMENT_FILE \
    $GLOBAL_INDEX_PREFIX \
    $MAX_NORM_FILE



