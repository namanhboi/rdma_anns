#!/bin/bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: ${BASH_SOURCE[0]} <source_index_path_prefix> <output_index_path_prefix> <partition_id>"
fi

SOURCE_INDEX_PATH_PREFIX=$1
OUTPUT_INDEX_PATH_PREFIX=$2
PARTITION_ID=$3


PARTITION_INDEX_PREFIX="${OUTPUT_INDEX_PATH_PREFIX}_partition${PARTITION_ID}"



ln -sf "${OUTPUT_INDEX_PATH_PREFIX}_partition_assignment.bin" "${PARTITION_INDEX_PREFIX}_partition_assignment.bin"


# PQ data

ln -sf "${SOURCE_INDEX_PATH_PREFIX}_pq_pivots.bin" "${PARTITION_INDEX_PREFIX}_pq_pivots.bin"


ln -sf "${SOURCE_INDEX_PATH_PREFIX}_pq_compressed.bin" "${PARTITION_INDEX_PREFIX}_pq_compressed.bin"


# mem index data
ln -sf "${SOURCE_INDEX_PATH_PREFIX}_mem.index" "${PARTITION_INDEX_PREFIX}_mem.index"
ln -sf "${SOURCE_INDEX_PATH_PREFIX}_mem.index.data" "${PARTITION_INDEX_PREFIX}_mem.index.data"
ln -sf "${SOURCE_INDEX_PATH_PREFIX}_mem.index.tags" "${PARTITION_INDEX_PREFIX}_mem.index.tags"


