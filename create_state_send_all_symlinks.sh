#!/bin/bash

set -euo pipefail

# Check if all arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <source_index_path_prefix> <output_index_path_prefix> <num_partitions>"
    exit 1
fi

SOURCE_INDEX_PATH_PREFIX=$1
OUTPUT_INDEX_PATH_PREFIX=$2
NUM_PARTITIONS=$3


# Check if num_partitions is a positive integer
if ! [[ "$NUM_PARTITIONS" =~ ^[0-9]+$ ]] || [ "$NUM_PARTITIONS" -lt 1 ]; then
    echo "Error: Number of partitions must be a positive integer!"
    exit 1
fi

for ((i = 0; i < NUM_PARTITIONS; i++)); do
    PARTITION_INDEX_PREFIX="${OUTPUT_INDEX_PATH_PREFIX}_partition${i}"
    ln -s "${OUTPUT_INDEX_PATH_PREFIX}_partition_assignment.bin" "${PARTITION_INDEX_PREFIX}_partition_assignment.bin"
    # PQ data
    ln -s "${SOURCE_INDEX_PATH_PREFIX}_pq_pivots.bin" "${PARTITION_INDEX_PREFIX}_pq_pivots.bin"
    ln -s "${SOURCE_INDEX_PATH_PREFIX}_pq_compressed.bin" "${PARTITION_INDEX_PREFIX}_pq_compressed.bin"
    # mem index data
    ln -s "${SOURCE_INDEX_PATH_PREFIX}_mem.index" "${PARTITION_INDEX_PREFIX}_mem.index"
    ln -s "${SOURCE_INDEX_PATH_PREFIX}_mem.index.data" "${PARTITION_INDEX_PREFIX}_mem.index.data"
    ln -s "${SOURCE_INDEX_PATH_PREFIX}_mem.index.tags" "${PARTITION_INDEX_PREFIX}_mem.index.tags"
done


