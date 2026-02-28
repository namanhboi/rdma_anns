#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common parameters
DATASET_NAME="text2image1B"
DATASET_SIZE="1M"

MODE="local"
NUM_SEARCH_THREADS=4
MAX_BATCH_SIZE=8
OVERLAP=false
NUM_CLIENT_THREADS=1
USE_COUNTER_THREAD=false
USE_LOGGING=false
WRITE_QUERY_CSV=false
SEND_RATE=0
BASE_EXPERIMENT_NAME=local_qps_recall_${DATASET_NAME}_${DATASET_SIZE}
NUM_QUERIES_TO_SEND=1000000
# LVEC="400"
LVEC="10 15 20 25 30 35 40 50 60 80 120 200 400"
MEM_L=10
K_VALUE=10

# Helper function to run experiment with sleep
run_with_sleep() {
    $SCRIPT_DIR/../run_experiment.sh "$@"
    sleep 5
}

echo "WRITE_QUERY_CSV is ${WRITE_QUERY_CSV}"


# STATE_SEND experiments
# for SEND_RATE in 0; do 
#     for NUM_SERVERS in 2; do
# 	for BEAM_WIDTH in 8; do
#             EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}"
#             run_with_sleep "${EXPERIMENT_NAME}" \
# 			   "${NUM_SERVERS}" \
# 			   "${DATASET_NAME}" \
# 			   "${DATASET_SIZE}" \
# 			   "STATE_SEND" \
# 			   "${MODE}" \
# 			   "${NUM_SEARCH_THREADS}" \
# 			   "${MAX_BATCH_SIZE}" \
# 			   "${OVERLAP}" \
# 			   "${BEAM_WIDTH}" \
# 			   "${NUM_CLIENT_THREADS}" \
# 			   "${USE_COUNTER_THREAD}" \
# 			   "${USE_LOGGING}" \
# 			   "${SEND_RATE}" \
# 			   "${WRITE_QUERY_CSV}"
# 	done
#     done
# done

# DIST_SEARCH_MODE_LIST=("STATE_SEND" "STATE_SEND_CLIENT_GATHER")
# DIST_SEARCH_MODE_LIST=("STATE_SEND_CLIENT_GATHER")
# DIST_SEARCH_MODE_LIST=("STATE_SEND")

# DIST_SEARCH_MODE_LIST=("SCATTER_GATHER")
DIST_SEARCH_MODE_LIST=("SCATTER_GATHER_TOP_N")

SEND_RATE_LIST=(0)
TOP_N=9999999

for DIST_SEARCH_MODE in "${DIST_SEARCH_MODE_LIST[@]}"; do
    echo "dist search mode is $DIST_SEARCH_MODE"
    if [[ $DIST_SEARCH_MODE == "SINGLE_SERVER" ]]; then
	NUM_SERVERS_LIST=(1)
    else
	NUM_SERVERS_LIST=(2)
    fi

    if [[ "$DIST_SEARCH_MODE" == "SCATTER_GATHER_TOP_N" ]]; then
	TOP_N=1
    fi
    
    
    for SEND_RATE in ${SEND_RATE_LIST[@]}; do 
	for NUM_SERVERS in ${NUM_SERVERS_LIST[@]}; do
	    echo "num servers is $NUM_SERVERS"
	    for BEAM_WIDTH in 8; do
		EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}"
		run_with_sleep "${EXPERIMENT_NAME}" \
			       "${NUM_SERVERS}" \
			       "${DATASET_NAME}" \
			       "${DATASET_SIZE}" \
			       "$DIST_SEARCH_MODE" \
			       "${MODE}" \
			       "${NUM_SEARCH_THREADS}" \
			       "${MAX_BATCH_SIZE}" \
			       "${OVERLAP}" \
			       "${BEAM_WIDTH}" \
			       "${NUM_CLIENT_THREADS}" \
			       "${USE_COUNTER_THREAD}" \
			       "${USE_LOGGING}" \
			       "${SEND_RATE}" \
			       "${WRITE_QUERY_CSV}" \
			       $NUM_QUERIES_TO_SEND \
			       $MEM_L \
			       $K_VALUE \
			       $TOP_N \
			       $LVEC
	    done
	done
    done
done
