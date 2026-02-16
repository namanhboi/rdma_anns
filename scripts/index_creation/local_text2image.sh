SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo $SCRIPT_DIR

$SCRIPT_DIR/create_indices_v2.sh \
    text2image1B \
    1M \
    float \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M_partition0_ids_uint32.bin \
    /home/nam/big-ann-benchmarks/data/text2image1B/base.1B.fbin.crop_nb_1000000_data.normalized.bin \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/vamana_32_64_1.2 \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/clusters_2/ \
    32 \
    64 \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/global_partitions_2/ \
    local \
    mips \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M_partition_assignment.bin \
    /home/nam/big-ann-benchmarks/data/text2image1B/diskann_1M_disk.index_max_base_norm.bin
