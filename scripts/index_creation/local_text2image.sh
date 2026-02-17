SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo $SCRIPT_DIR

PARTITION_IDS=(0 1)



for i in "${PARTITION_IDS[@]}"
do
    $SCRIPT_DIR/create_indices_v2.sh \
    text2image1B \
    1M \
    float \
    "/home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M_partition${i}_ids_uint32.bin" \
    /home/nam/big-ann-benchmarks/data/text2image1B/base.1B.fbin.crop_nb_1000000_data.normalized.bin \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/vamana_32_64_1.0 \
    32 \
    64 \
    2 \
    local \
    mips \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M_partition_assignment.bin \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M \
    /home/nam/big-ann-benchmarks/data/text2image1B/1M/pipeann_1M \
    /home/nam/big-ann-benchmarks/data/text2image1B/diskann_1M_disk.index_max_base_norm.bin
done
