#!/bin/bash
# assuming that the data is in namanh@nfs:/mydata/local/anngraphs/{dataset_name}/{scale}
# this file will create the indices for both the scatter gather and state send
# approach from the partition file and put them in the specified folders
DATASET_NAME=$1
DATASET_SIZE=$2
DATA_TYPE=$3
PARTITION_FILE=$4
BASE_FILE=$5
PARLAYANN_GRAPH_FILE=$6
SCATTER_GATHER_OUTPUT=$7
SCATTER_GATHER_R=$8
SCATTER_GATHER_L=$9
STATE_SEND_OUTPUT=$10

if [ $# -ne 10 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <dataset_name> <dataset_size> <data_type> <partition_file> <base_file> <parlayann_graph_file> <scatter_gather_output> <scatter_gather_r> <scatter_gather_l> <state_send_output>"
    echo "  dataset_name: bigann"
    echo "  dataset_size: 10M or 100M or 1B"
    echo "  data_type: uint8 or int8 or float"
    echo "  partition_file: /mydata/local/anngraphs/bigann/1B/global_partitions_5/pipeann_1B_partition0_ids_uint32.bin"
    echo "  base_file: /mydata/local/anngraphs/bigann/1B/base.1B.u8bin"
    echo "  parlayann_graph_file: /mydata/local/anngraphs/bigann/1B/vamana_64_128_1.2"
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
SCATTER_GATHER_INDEX_PREFIX="${SCATTER_GATHER_OUTPUT}/pipeann_${DATASET_SIZE}"
STATE_SEND_INDEX_PREFIX="${STATE_SEND_OUTPUT}/pipeann_${DATASET_SIZE}"

[[ "$DATASET_NAME" != "bigann" ]] && { echo "Error: dataset_name must be 'bigann'"; exit 1; }
[[ "$DATASET_SIZE" != "100M" && "$DATASET_SIZE" != "1B" ]] && { echo "Error: dataset_size must be 100M or 1B"; exit 1; }

DATA_FOLDER="/mydata/local/anngraphs/${DATASET_NAME}/${DATASET_SIZE}/"
if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "${DATA_FOLDER} doesn't exist"
    exit 1
fi

# making directory to store all the bin files
PARTITION_BASE_FILE_FOLDER="${DATA_FOLDER}/base_files/"
mkdir -p "${PARTITION_BASE_FILE_FOLDER}"
PARTITION_BASE_FILE_PATH="${PARTITION_BASE_FILE_FOLDER}/pipeann_${DATASET_SIZE}_partition.bin"

WORKDIR="$HOME/workspace/rdma_anns/"

# first, we go about creating the base file in the scatter gather folder
"${WORKDIR}/build/src/state_send/create_base_file_from_loc_file" \
    "${DATA_TYPE}" \
    "${BASE_FILE}" \
    "${PARTITION_FILE}" \
    "${PARTITION_BASE_FILE_PATH}"

# now we need to create the pq file
"${WORKDIR}/build/src/state_send/create_pq_data" \
    "${DATA_TYPE}" \
    "${PARTITION_BASE_FILE_PATH}" \
    "${SCATTER_GATHER_INDEX_PREFIX}" \
    "${METRIC}" \
    "${SCATTER_GATHER_NUM_PQ_CHUNKS}"

# now we need to do create the mem index
SCATTER_GATHER_SLICE_PREFIX="${SCATTER_GATHER_INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
"${WORKDIR}/build/src/state_send/gen_random_slice" \
    "${PARTITION_BASE_FILE_PATH}" \
    "${SCATTER_GATHER_SLICE_PREFIX}" \
    "${MEM_INDEX_SAMPLING_RATE}"

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

# now we actually create the disk index for scatter gather
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

echo "Index creation complete!"
