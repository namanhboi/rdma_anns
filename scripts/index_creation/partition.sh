#!/usr/bin/bash


set -ueo pipefail


if [[ $# -ne 4 ]]; then
    echo "Usage: <metric> <base_file> <partition_file> <num_partitions>"
    exit 1
fi

METRIC=$1
BASE_FILE=$2
PARTITION_FILE=$3
NUM_PARTITIONS=$4
METHOD=GP

if [ $METRIC != "l2" -a $METRIC != "mips" ]; then
    echo "currently only supports l2 and mips as metrics"
    exit 1
fi

$HOME/workspace/rdma_anns/extern/gp-ann/build_${METRIC}/Partition $BASE_FILE ${PARTITION_FILE} $NUM_PARTITIONS $METHOD default



