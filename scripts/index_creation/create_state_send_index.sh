#!/usr/bin/bash
# create state send disk index 

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common_vars.sh



DATA_TYPE=$1
METRIC=$2
STATE_SEND_INDEX_PREFIX=$3
PARTITION_ID_FILE=$4
PARTITION_BASE_FILE=$5
PARTITION_GRAPH_FILE=$6
PARTITION_ASSIGNMENT_FILE=$7
GLOBAL_INDEX_PREFIX=$8
MAX_NORM_FILE=${9:-""}

if [[ ! -f $PARTITION_BASE_FILE ]]; then
    echo "partition base file doesnt exist: $PARTITION_BASE_FILE"
    exit 1
fi
if [[ ! -f $PARTITION_GRAPH_FILE ]]; then
    echo "partition graph file doesnt exist: $PARTITION_GRAPH_FILE"
    exit 1
fi

if [[ ! -f $PARTITION_ID_FILE ]]; then
    echo "partition id file doesnt exist: $PARTITION_ID_FILE"
    exit 1
fi


if [[ $METRIC == "mips" ]]; then
    if [[ $DATA_TYPE != "float" ]]; then
	echo "for mips, only float as data type is accepted"
	exit 1
    fi

    if [[ ! -f $MAX_NORM_FILE  ]]; then
	echo "for mips, max_norm_file ($MAX_NORM_FILE) for the partition_base_file has to be exist and be provided"
	exit 1
    fi

    if [[ $PARTITION_BASE_FILE != *"${NORMALIZED_SUFFIX}" ]]; then
	echo "for mips, the partition base file ${PARTITION_BASE_FILE} has to be normalized aka ending in ${NORMALIZED_SUFFIX}"
	exit 1
    fi
fi

STATE_SEND_PARTITION_ID_FILE=${STATE_SEND_INDEX_PREFIX}_ids_uint32.bin
ln -sf "$PARTITION_ID_FILE" "$STATE_SEND_PARTITION_ID_FILE"



MEM_INDEX_PATH=${GLOBAL_INDEX_PREFIX}_mem.index
PQ_COMPRESSED_PATH=${GLOBAL_INDEX_PREFIX}_pq_compressed.bin
PQ_PIVOT_PATH=${GLOBAL_INDEX_PREFIX}_pq_pivots.bin


if [[ ! -f $MEM_INDEX_PATH ]]; then
    echo "The global mem index doesn't exist : $MEM_INDEX_PATH"
    exit 1
fi

if [[ ! -f $PQ_PIVOT_PATH ]]; then
    echo "The global pq pivot file doesn't exist : $PQ_PIVOT_PATH"
    exit 1
fi

if [[ ! -f $PQ_COMPRESSED_PATH ]]; then
    echo "The global pq comprssed file doesn't exist : $PQ_COMPRESSED_PATH"
    exit 1
fi



if [[ ! -f "${STATE_SEND_INDEX_PREFIX}_disk.index" ]]; then
"${WORKDIR}/build/src/state_send/build_disk_index_from_bin_graph" \
    "${DATA_TYPE}" \
    "${PARTITION_BASE_FILE}" \
    "${PARTITION_GRAPH_FILE}" \
    "${STATE_SEND_INDEX_PREFIX}_disk.index"
fi




PARTITION_STATE_SEND_MEM_INDEX_PATH="${STATE_SEND_INDEX_PREFIX}_mem.index"
ln -sf "${MEM_INDEX_PATH}" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}"
ln -sf "${MEM_INDEX_PATH}.tags" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.tags"
ln -sf "${MEM_INDEX_PATH}.data" "${PARTITION_STATE_SEND_MEM_INDEX_PATH}.data"        
echo "Symlinked existing memory index"



PARTITION_STATE_SEND_PQ_COMPRESSED_PATH="${STATE_SEND_INDEX_PREFIX}_pq_compressed.bin"
PARTITION_STATE_SEND_PQ_PIVOT_PATH="${STATE_SEND_INDEX_PREFIX}_pq_pivots.bin"
ln -sf "${PQ_COMPRESSED_PATH}" "${PARTITION_STATE_SEND_PQ_COMPRESSED_PATH}"
ln -sf "${PQ_PIVOT_PATH}" "${PARTITION_STATE_SEND_PQ_PIVOT_PATH}"
echo "Symlinked existing PQ files"


STATE_SEND_PARTITION_ASSIGNMENT_PATH="${STATE_SEND_INDEX_PREFIX}_partition_assignment.bin"
ln -sf ${PARTITION_ASSIGNMENT_FILE} ${STATE_SEND_PARTITION_ASSIGNMENT_PATH}

STATE_SEND_MAX_NORM_FILE="${STATE_SEND_INDEX_PREFIX}_disk.index_max_base_norm.bin"
if [[ -f $MAX_NORM_FILE && (! -f $STATE_SEND_MAX_NORM_FILE) ]]; then
    ln -sf ${MAX_NORM_FILE} ${STATE_SEND_MAX_NORM_FILE}
fi

echo "State send index complete"
