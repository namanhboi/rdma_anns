#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common parameters
DATASET_NAME="bigann"
DATASET_SIZE="100M"

MODE="distributed"
NUM_SEARCH_THREADS=8
MAX_BATCH_SIZE=8
OVERLAP=false
NUM_CLIENT_THREADS=1
USE_COUNTER_THREAD=false
USE_LOGGING=false
SEND_RATE=0
BASE_EXPERIMENT_NAME=final_with_inter_long_lvec_bigann_${DATASET_SIZE}
# Helper function to run experiment with sleep
run_with_sleep() {
    $SCRIPT_DIR/run_experiment.sh "$@"
    sleep 10
}

# SCATTER_GATHER experiments 
for SEND_RATE in 0; do
    for NUM_SERVERS in 5 10; do
	for BEAM_WIDTH in 8; do
            EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}_send_${SEND_RATE}"
            run_with_sleep "${EXPERIMENT_NAME}" \
			   "${NUM_SERVERS}" \
			   "${DATASET_NAME}" \
			   "${DATASET_SIZE}" \
			   "SCATTER_GATHER" \
			   "${MODE}" \
			   "${NUM_SEARCH_THREADS}" \
			   "${MAX_BATCH_SIZE}" \
			   "${OVERLAP}" \
			   "${BEAM_WIDTH}" \
			   "${NUM_CLIENT_THREADS}" \
			   "${USE_COUNTER_THREAD}" \
			   "${USE_LOGGING}" \
			   "${SEND_RATE}"
	done
    done
done

# STATE_SEND experiments
for SEND_RATE in 0; do 
    for NUM_SERVERS in 5 10; do
	for BEAM_WIDTH in 8; do
            EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}_send_${SEND_RATE}"
            run_with_sleep "${EXPERIMENT_NAME}" \
			   "${NUM_SERVERS}" \
			   "${DATASET_NAME}" \
			   "${DATASET_SIZE}" \
			   "STATE_SEND" \
			   "${MODE}" \
			   "${NUM_SEARCH_THREADS}" \
			   "${MAX_BATCH_SIZE}" \
			   "${OVERLAP}" \
			   "${BEAM_WIDTH}" \
			   "${NUM_CLIENT_THREADS}" \
			   "${USE_COUNTER_THREAD}" \
			   "${USE_LOGGING}" \
			   "${SEND_RATE}"
	done
    done
done



# SINGLE_SERVER experiments with varying search thread counts
# for NUM_SEARCH_THREADS in 8; do
#     for BEAM_WIDTH in 8; do
#         EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_beam_${BEAM_WIDTH}"
#         run_with_sleep "${EXPERIMENT_NAME}" \
#                        1 \
#                        "${DATASET_NAME}" \
#                        "${DATASET_SIZE}" \
#                        "SINGLE_SERVER" \
#                        "${MODE}" \
#                        "${NUM_SEARCH_THREADS}" \
#                        "${MAX_BATCH_SIZE}" \
#                        "${OVERLAP}" \
#                        "${BEAM_WIDTH}" \
#                        "${NUM_CLIENT_THREADS}" \
#                        "${USE_COUNTER_THREAD}" \
#                        "${USE_LOGGING}" \
#                        "${SEND_RATE}"
#     done
# done
