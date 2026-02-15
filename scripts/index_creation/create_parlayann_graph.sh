#!/bin/bash
# creates index with parlayann

set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: <R> <L> <data_type> <dist_fn> <base_file> <graph_file>"
    exit 1
fi

R=$1
L=$2
DATA_TYPE=$3
DIST_FN=$4
BASE_FILE=$5
GRAPH_FILE=$6


ALPHA=1.2
TWO_PASS=0

$HOME/workspace/rdma_anns/extern/ParlayANN/algorithms/vamana/neighbors -R $R -L $L -alpha $ALPHA two_pass $TWO_PASS -graph_outfile $GRAPH_FILE -data_type $DATA_TYPE -dist_func $DIST_FN -base_path $BASE_FILE
