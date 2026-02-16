#!/usr/bin/bash
# create scatter gather disk index + mem index (with same r, l) based on sampling rate

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common_vars.sh

if [[ $# -lt 11 ]]; then
    echo "Usage: <data_type> <metric> <R> <L> <scatter_gather_index_prefix> <partition_id_file> <partition_base_file> <max_norm_file(optional)>"
    echo "  data_type: uint8, int8, float"
    echo "  metric: mips, l2"
    echo "  R: max nbr. 32 for small scale, 64 for 100M or 1B"
    echo "  L"
    echo "  scatter_gather_index_prefix: prefix to output all the necessary files"
    echo "  partition_id_file: partition id file which is used as tag file to map back to original ids"
    echo "  partition_base_file: partition base file used to create this scatter gather index. Is cut from original base file. Need to be normalized for mips "
    echo "  max_norm_file: max norm of base file (only needed for mips)"
fi

DATA_TYPE=$1
METRIC=$2
R=$3
L=$4
NUM_PQ_CHUNKS=$5
RAM_BUDGET=$6
NUM_THREADS=$7
SCATTER_GATHER_INDEX_PREFIX=$8
MEM_INDEX_SAMPLING_RATE=$9
PARTITION_ID_FILE=${10}
PARTITION_BASE_FILE=${11}
MAX_NORM_FILE=${12:-""}

if [[ ! -f $PARTITION_BASE_FILE ]]; then
    echo "$PARTITION_BASE_FILE doesn't exist"
    exit 1
fi


if [[ ! -f $PARTITION_ID_FILE ]]; then
    echo "$PARTITION_ID_FILE doesn't exist"
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

# build the actual disk index
if [[ ! -f "${SCATTER_GATHER_INDEX_PREFIX}_disk.index" ]]; then
    "${WORKDIR}/build/src/state_send/build_disk_index" \
	$DATA_TYPE \
	$PARTITION_BASE_FILE \
	$SCATTER_GATHER_INDEX_PREFIX \
	$R \
	$L \
	$RAM_BUDGET \
	$NUM_PQ_CHUNKS \
	$NUM_THREADS \
	$METRIC \
	0
fi


# now build in mem index
if [[ $MEM_INDEX_SAMPLING_RATE != "0" || $MEM_INDEX_SAMPLING_RATE != "0.0" || $MEM_INDEX_SAMPLING_RATE != "0.00" ]]; then
    MEM_INDEX_ALPHA=1.2
    SCATTER_GATHER_SLICE_PREFIX="${SCATTER_GATHER_INDEX_PREFIX}_SAMPLE_RATE_${MEM_INDEX_SAMPLING_RATE}"
    SCATTER_GATHER_SLICE_TAG="${SCATTER_GATHER_SLICE_PREFIX}_ids.bin"
    if [[ $METRIC == "mips" ]]; then
	SCATTER_GATHER_SLICE_DATA="${SCATTER_GATHER_SLICE_PREFIX}${NORMALIZED_SUFFIX}"
    else
	SCATTER_GATHER_SLICE_DATA="${SCATTER_GATHER_SLICE_PREFIX}_data.bin"
    fi
    echo "SCATTER_GATHER_SLICE_DATA is $SCATTER_GATHER_SLICE_DATA"
    
    if [[ (! -f $SCATTER_GATHER_SLICE_DATA) && (! -f $SCATTER_GATHER_SLICE_TAG) ]]; then
	"${WORKDIR}/build/src/state_send/gen_random_slice" \
	    "${DATA_TYPE}" \
	    "${PARTITION_BASE_FILE}" \
	    "${SCATTER_GATHER_SLICE_DATA}" \
	    "${MEM_INDEX_SAMPLING_RATE}"
    fi

    # need to double check build_mem_index.cpp
    if [[ ! -f "${SCATTER_GATHER_INDEX_PREFIX}_mem.index" ]]; then
	"${WORKDIR}/build/src/state_send/build_memory_index" \
	    "${DATA_TYPE}" \
	    "${SCATTER_GATHER_SLICE_DATA}" \
	    "${SCATTER_GATHER_SLICE_TAG}" \
	    "${R}" \
	    "${L}" \
	    "${MEM_INDEX_ALPHA}" \
	    "${SCATTER_GATHER_INDEX_PREFIX}_mem.index" \
	    "${NUM_THREADS}" \
	    "${METRIC}"
    fi
fi


# now we need to symlink the tag and max norm file (if it exists)
SCATTER_GATHER_TAG_FILE="${SCATTER_GATHER_INDEX_PREFIX}_disk.index.tags"
if [[ ! -f ${SCATTER_GATHER_TAG_FILE} ]]; then
    ln -sf ${PARTITION_ID_FILE} ${SCATTER_GATHER_TAG_FILE}
fi


SCATTER_GATHER_MAX_NORM_FILE="${SCATTER_GATHER_INDEX_PREFIX}_disk.index_max_base_norm.bin"
if [[ (-f $MAX_NORM_FILE) && (! -f $SCATTER_GATHER_MAX_NORM_FILE)]]; then
    ln -sf "$MAX_NORM_FILE" "$SCATTER_GATHER_MAX_NORM_FILE"
fi


echo "Scatter-gather index creation complete!"

   


