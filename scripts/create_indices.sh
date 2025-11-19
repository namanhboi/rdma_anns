#!/bin/bash

# assuming that the data is in namanh@nfs:/mydata/local/anngraphs/{dataset_name}/{scale}
# this file will create the indices for both the scatter gather and state send
# approach from the partition file and put them in the specified folders

set -euxo pipefail

DATASET_NAME=$1
DATASET_SIZE=$2
DATA_TYPE=$3
PARTITION_FILE=$4
BASE_FILE=$5
GRAPH_FILE=$6
SCATTER_GATHER_OUTPUT=$7
SCATTER_GATHER_R=$8
SCATTER_GATHER_L=$9
STATE_SEND_OUTPUT=${10}

if [ $# -ne 10 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <dataset_name> <dataset_size> <data_type> <partition_file> <base_file> <graph_file> <scatter_gather_output> <scatter_gather_r> <scatter_gather_l> <state_send_output>"
    echo "  dataset_name: bigann"
    echo "  dataset_size: 10M or 100M or 1B"
    echo "  data_type: uint8 or int8 or float"
    echo "  partition_file: /mydata/local/anngraphs/bigann/1B/global_partitions_5/pipeann_1B_partition0_ids_uint32.bin"
    echo "  base_file: /mydata/local/anngraphs/bigann/1B/base.1B.u8bin"
    echo "  graph_file: /mydata/local/anngraphs/bigann/1B/vamana_64_128_1.2"
    echo "  scatter_gather_output: /mydata/local/anngraphs/bigann/1B/clusters_5/"
    echo "  scatter_gather_r: 64"
    echo "  scatter_gather_l: 128"
    echo "  state_send_output: /mydata/local/anngraphs/bigann/1B/global_partitions_5/"
    exit 1
fi

NUM_THREADS=56
METRIC=l2
MEM_INDEX_SAMPLING_RATE=0.01
MEM_INDEX_R=32
MEM_INDEX_L=64
MEM_INDEX_ALPHA=1.2
SCATTER_GATHER_ALPHA=1.2
SCATTER_GATHER_NUM_PQ_CHUNKS=32


[[ "$DATASET_NAME" != "bigann" ]] && { echo "Error: dataset_name must be 'bigann'"; exit 1; }
[[ "$DATASET_SIZE" != "100M" && "$DATASET_SIZE" != "1B" ]] && { echo "Error: dataset_size must be 100M or 1B"; exit 1; }

DATA_FOLDER="/mydata/local/anngraphs/${DATASET_NAME}/${DATASET_SIZE}/"
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

if [[ ! -f "$PARTITION_FILE" ]]; then
    echo "${PARTITION_FILE} doesn't exist"
    exit 1
fi



# Extract partition number early - we'll need it later
filename=$(basename "$PARTITION_FILE" .bin)
if [[ "$filename" =~ partition([0-9]+) ]]; then
    PARTITION_NUM="${BASH_REMATCH[1]}"
    PARTITION_ID="partition${PARTITION_NUM}"
    echo "Processing partition: $PARTITION_ID (number: $PARTITION_NUM)"
else
    echo "Error: Could not extract partition number from $filename"
    exit 1
fi

dirname=$(basename "$STATE_SEND_OUTPUT")
# NUM_SERVERS=${dirname##*_}
# NUM_SERVERS=${num%/}


STATE_SEND_INDEX_PREFIX="${STATE_SEND_OUTPUT}/pipeann_${DATASET_SIZE}_${PARTITION_ID}"
SCATTER_GATHER_INDEX_PREFIX="${SCATTER_GATHER_OUTPUT}/pipeann_${DATASET_SIZE}_cluster${PARTITION_NUM}"

# making directory to store all the bin files
PARTITION_BASE_FILE_FOLDER="${DATA_FOLDER}/base_files/${dirname}"
mkdir -p "${PARTITION_BASE_FILE_FOLDER}"
PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_FOLDER}/pipeann_${DATASET_SIZE}_${PARTITION_ID}.bin"

WORKDIR="$HOME/workspace/rdma_anns/"

# first, we go about creating the base file in the scatter gather folder
if [[ ! -f "${PARTITION_BASE_FILE_PATH}" ]]; then 
"${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
    "${DATA_TYPE}" \
    "${BASE_FILE}" \
    "${PARTITION_FILE}" \
    "${PARTITION_BASE_FILE_PATH}"
fi

# now we need to create the pq file
if [[ (! -f "${SCATTER_GATHER_INDEX_PREFIX}_pq_compressed.bin") || (! -f "${SCATTER_GATHER_INDEX_PREFIX}_pq_pivots.bin") ]]; then
    "${WORKDIR}/build/src/state_send/create_pq_data" \
	"${DATA_TYPE}" \
	"${PARTITION_BASE_FILE_PATH}" \
	"${SCATTER_GATHER_INDEX_PREFIX}" \
	"${METRIC}" \
	"${SCATTER_GATHER_NUM_PQ_CHUNKS}"
fi

# now we need to do create the mem index
SCATTER_GATHER_SLICE_PREFIX="${SCATTER_GATHER_INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
if [[ (! -f "${SCATTER_GATHER_SLICE_PREFIX}_data.bin") && (! -f "${SCATTER_GATHER_SLICE_PREFIX}_ids.bin") ]]; then
    "${WORKDIR}/build/src/state_send/gen_random_slice" \
	"${DATA_TYPE}" \
	"${PARTITION_BASE_FILE_PATH}" \
	"${SCATTER_GATHER_SLICE_PREFIX}" \
	"${MEM_INDEX_SAMPLING_RATE}"
fi

if [[ ! -f "${SCATTER_GATHER_INDEX_PREFIX}_mem.index" ]]; then
    "${WORKDIR}/build/src/state_send/build_memory_index" \
	"${DATA_TYPE}" \
	"${SCATTER_GATHER_SLICE_PREFIX}_data.bin" \
	"${SCATTER_GATHER_SLICE_PREFIX}_ids.bin" \
	"${MEM_INDEX_R}" \
	"${MEM_INDEX_L}" \
	"${MEM_INDEX_ALPHA}" \
	"${SCATTER_GATHER_INDEX_PREFIX}_mem.index" \
	"${NUM_THREADS}" \
	"${METRIC}"
fi

# now we actually create the disk index for scatter gather
if [[ ! -f "${SCATTER_GATHER_INDEX_PREFIX}_disk.index" ]]; then
    RAM_BUDGET=96
    "${WORKDIR}/build/src/state_send/build_disk_index" \
	"${DATA_TYPE}" \
	"${PARTITION_BASE_FILE_PATH}" \
	"${SCATTER_GATHER_INDEX_PREFIX}" \
	"${SCATTER_GATHER_R}" \
	"${SCATTER_GATHER_L}" \
	3.3 \
	"${RAM_BUDGET}" \
	"${NUM_THREADS}" \
	"${METRIC}" \
	0
fi


PARTITION_SCATTER_GATHER_TAG_FILE="${SCATTER_GATHER_INDEX_PREFIX}_disk.index.tags"
if [[ ! -f ${PARTITION_SCATTER_GATHER_TAG_FILE} ]]; then
    ln -sf ${PARTITION_FILE} ${PARTITION_SCATTER_GATHER_TAG_FILE}
fi

echo "Scatter-gather index creation complete!"

# Now create STATE_SEND indices
PARTITION_STATE_SEND_GRAPH_FOLDER="${DATA_FOLDER}/graph_files/${dirname}"
mkdir -p "${PARTITION_STATE_SEND_GRAPH_FOLDER}"
PARTITION_STATE_SEND_GRAPH_FILE="${PARTITION_STATE_SEND_GRAPH_FOLDER}/pipeann_${DATASET_SIZE}_${PARTITION_ID}_graph"


if [[ ! -f "${PARTITION_STATE_SEND_GRAPH_FILE}" ]]; then
    "${WORKDIR}/build/src/state_send/create_partition_graph_file" \
	"${GRAPH_FILE}" \
	"${PARTITION_FILE}" \
	"${PARTITION_STATE_SEND_GRAPH_FILE}"
fi


if [[ ! -f "${STATE_SEND_INDEX_PREFIX}_disk.index" ]]; then
"${WORKDIR}/build/src/state_send/build_disk_index_from_bin_graph" \
    "${DATA_TYPE}" \
    "${PARTITION_BASE_FILE_PATH}" \
    "${PARTITION_STATE_SEND_GRAPH_FILE}" \
    "${STATE_SEND_INDEX_PREFIX}_disk.index"
fi

# Handle memory index - check if global one exists, otherwise create it
INDEX_PREFIX="${DATA_FOLDER}/pipeann_${DATASET_SIZE}"
MEM_INDEX_PATH="${INDEX_PREFIX}_mem.index"
PARTITION_STATE_SEND_MEM_INDEX_PATH="${STATE_SEND_INDEX_PREFIX}_mem.index"

if [[ -f "${MEM_INDEX_PATH}" ]]; then
    # symlink to existing global mem index
    ln -sf "${MEM_INDEX_PATH}" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}"
    ln -sf "${MEM_INDEX_PATH}.tags" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.tags"
    ln -sf "${MEM_INDEX_PATH}.data" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.data"        
    echo "Symlinked existing memory index"
else
    # create the global mem index
    echo "Creating new memory index..."
    SLICE_PREFIX="${INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
    "${WORKDIR}/build/src/state_send/gen_random_slice" \
	"${DATA_TYPE}"
        "${BASE_FILE}" \
        "${SLICE_PREFIX}" \
        "${MEM_INDEX_SAMPLING_RATE}"

    "${WORKDIR}/build/src/state_send/build_memory_index" \
        "${DATA_TYPE}" \
        "${SLICE_PREFIX}_data.bin" \
        "${SLICE_PREFIX}_ids.bin" \
        "${MEM_INDEX_R}" \
        "${MEM_INDEX_L}" \
        "${MEM_INDEX_ALPHA}" \
        "${MEM_INDEX_PATH}" \
        "${NUM_THREADS}" \
        "${METRIC}"
    
    # Now symlink it
    ln -sf "${MEM_INDEX_PATH}" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}"
    ln -sf "${MEM_INDEX_PATH}.tags" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.tags"
    ln -sf "${MEM_INDEX_PATH}.data" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.data"
fi

# Handle PQ files - check if global ones exist, otherwise create them
PQ_COMPRESSED_PATH="${INDEX_PREFIX}_pq_compressed.bin"
PQ_PIVOT_PATH="${INDEX_PREFIX}_pq_pivots.bin"
PARTITION_STATE_SEND_PQ_COMPRESSED_PATH="${STATE_SEND_INDEX_PREFIX}_pq_compressed.bin"
PARTITION_STATE_SEND_PQ_PIVOT_PATH="${STATE_SEND_INDEX_PREFIX}_pq_pivots.bin"

if [[ -f "${PQ_COMPRESSED_PATH}" && -f "${PQ_PIVOT_PATH}" ]]; then
    # symlink both pq files
    ln -sf "${PQ_COMPRESSED_PATH}" "${PARTITION_STATE_SEND_PQ_COMPRESSED_PATH}"
    ln -sf "${PQ_PIVOT_PATH}" "${PARTITION_STATE_SEND_PQ_PIVOT_PATH}"
    echo "Symlinked existing PQ files"
else
    # create the pq files
    echo "Creating new PQ files..."
    "${WORKDIR}/build/src/state_send/create_pq_data" \
        "${DATA_TYPE}" \
        "${BASE_FILE}" \
        "${INDEX_PREFIX}" \
        "${METRIC}" \
        "${SCATTER_GATHER_NUM_PQ_CHUNKS}"
    
    # Now symlink them
    ln -sf "${PQ_COMPRESSED_PATH}" "${PARTITION_STATE_SEND_PQ_COMPRESSED_PATH}"
    ln -sf "${PQ_PIVOT_PATH}" "${PARTITION_STATE_SEND_PQ_PIVOT_PATH}"
fi


PARTITION_ASSIGNMENT_PATH="${STATE_SEND_OUTPUT}/pipeann_${DATASET_SIZE}_partition_assignment.bin"
PARTITION_STATE_SEND_PARTITION_ASSIGNMENT_PATH="${STATE_SEND_INDEX_PREFIX}_partition_assignment.bin"
if [[ -f ${PARTITION_ASSIGNMENT_PATH} ]]; then
    ln -sf ${PARTITION_ASSIGNMENT_PATH} ${PARTITION_STATE_SEND_PARTITION_ASSIGNMENT_PATH}
else
    echo "file doesn't exist, need to cp from nfs: ${PARTITION_ASSIGNMENT_PATH}"
    exit 1
fi


echo "All index creation complete!"
