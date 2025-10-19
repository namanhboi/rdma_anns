#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_folder> <target_folder>"
    echo "Example: $0 ./ ./test_folder"
    exit 1
fi

SOURCE_FOLDER="$1"
TARGET_FOLDER="$2"

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
echo "Creating symlinks from $SOURCE_FOLDER to $TARGET_FOLDER..."

ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index" "$TARGET_FOLDER/pipeann_10M_partition0_mem_index"
ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index" "$TARGET_FOLDER/pipeann_10M_partition1_mem_index"
ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.data" "$TARGET_FOLDER/pipeann_10M_partition0_mem_index.data"
ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.data" "$TARGET_FOLDER/pipeann_10M_partition1_mem_index.data"
ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.tags" "$TARGET_FOLDER/pipeann_10M_partition0_mem_index.tags"
ln -s "$SOURCE_FOLDER/pipeann_10M_mem.index.tags" "$TARGET_FOLDER/pipeann_10M_partition1_mem_index.tags"
ln -s "$SOURCE_FOLDER/pipeann_10M_pq_compressed.bin" "$TARGET_FOLDER/pipeann_10M_partition0_pq_compressed.bin"
ln -s "$SOURCE_FOLDER/pipeann_10M_pq_compressed.bin" "$TARGET_FOLDER/pipeann_10M_partition1_pq_compressed.bin"
ln -s "$SOURCE_FOLDER/pipeann_10M_pq_pivots.bin" "$TARGET_FOLDER/pipeann_10M_partition0_pq_pivots.bin"
ln -s "$SOURCE_FOLDER/pipeann_10M_pq_pivots.bin" "$TARGET_FOLDER/pipeann_10M_partition1_pq_pivots.bin"

echo "Symlinks created successfully!"
