#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_EXPERIMENT_NAME=final_with_inter_deep_100M_qps_recall

# Common parameters
DATASET_NAME="deep1b"
DATASET_SIZE="100M"
MODE="distributed"
NUM_SEARCH_THREADS=8
MAX_BATCH_SIZE=8
OVERLAP=false
NUM_CLIENT_THREADS=1
USE_COUNTER_THREAD=false
USE_LOGGING=false
SEND_RATE=0

# Helper function to run experiment with sleep
run_with_sleep() {
    $SCRIPT_DIR/run_experiment.sh "$@"
    sleep 10
}

# SCATTER_GATHER experiments
for NUM_SERVERS in 10 5; do
    for BEAM_WIDTH in 8; do
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}"
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

# STATE_SEND experiments
for NUM_SERVERS in 10 5; do
    for BEAM_WIDTH in 8; do
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}"
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


for NUM_SEARCH_THREADS in 8 16 32; do
    for BEAM_WIDTH in 8; do
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_beam_${BEAM_WIDTH}"
        run_with_sleep "${EXPERIMENT_NAME}" \
                       1 \
                       "${DATASET_NAME}" \
                       "${DATASET_SIZE}" \
                       "SINGLE_SERVER" \
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
