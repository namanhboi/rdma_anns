#!/bin/bash

# Check if all arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <source_folder> <target_folder> <num_partitions>"
    echo "Example: $0 ./ ./test_folder 2"
    exit 1
fi

SOURCE_FOLDER="$1"
TARGET_FOLDER="$2"
NUM_PARTITIONS="$3"

# Check if num_partitions is a positive integer
if ! [[ "$NUM_PARTITIONS" =~ ^[0-9]+$ ]] || [ "$NUM_PARTITIONS" -lt 1 ]; then
    echo "Error: Number of partitions must be a positive integer!"
    exit 1
fi

# Check if source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder '$SOURCE_FOLDER' does not exist!"
    exit 1
fi

# Check if target folder exists
if [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: Target folder '$TARGET_FOLDER' does not exist!"
    exit 1
fi

# Create symlinks
echo "Creating symlinks from $SOURCE_FOLDER to $TARGET_FOLDER for $NUM_PARTITIONS partitions..."

for ((i=0; i<NUM_PARTITIONS; i++)); do
    ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index" "$TARGET_FOLDER/pipeann_10M_partition${i}_mem.index"
    ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.data" "$TARGET_FOLDER/pipeann_10M_partition${i}_mem.index.data"
    ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.tags" "$TARGET_FOLDER/pipeann_10M_partition${i}_mem.index.tags"
    ln -s "$SOURCE_FOLDER/pipeann_10M_pq_compressed.bin" "$TARGET_FOLDER/pipeann_10M_partition${i}_pq_compressed.bin"
    ln -s "$SOURCE_FOLDER/pipeann_10M_pq_pivots.bin" "$TARGET_FOLDER/pipeann_10M_partition${i}_pq_pivots.bin"
done

echo "Symlinks created successfully!"
