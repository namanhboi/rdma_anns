#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common parameters
DATASET_NAME="text2image1B"
DATASET_SIZE="1M"

MODE="local"
NUM_SEARCH_THREADS=4
NUM_ORCHESTRATION_THREADS=4
NUM_SCORING_THREADS=4
MAX_BATCH_SIZE=8
OVERLAP=false
NUM_CLIENT_THREADS=1
USE_COUNTER_THREAD=false
USE_LOGGING=false
WRITE_QUERY_CSV=false
SEND_RATE=0
MEM_K=10
MEM_L=10

NUM_QUERIES_TO_SEND=1000000
# LVEC="400"
LVEC="10 15 20 25"



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

# DIST_SEARCH_MODE_LIST=("DISTRIBUTED_ANN")
# DIST_SEARCH_MODE_LIST=("DISTRIBUTED_ANN")
DIST_SEARCH_MODE_LIST=("STATE_SEND")

# DIST_SEARCH_MODE_LIST=("SCATTER_GATHER" "STATE_SEND" "DISTRIBUTED_ANN")
# DIST_SEARCH_MODE_LIST=("SCATTER_GATHER_TOP_N")
# DIST_SEARCH_MODE_LIST=("SINGLE_SERVER")
# DIST_SEARCH_MODE_LIST=("DISTRIBUTED_ANN")
SEND_RATE_LIST=(0)
TOP_N=9999999
BEAM_WIDTH_LIST=()
SEARCH_THREAD_MODE_LIST=("BATANN")
K_VALUE_LIST=(10)
for i in {1..1}; do
    for SEARCH_THREAD_MODE in "${SEARCH_THREAD_MODE_LIST[@]}"; do 
	for K_VALUE in "${K_VALUE_LIST[@]}"; do
	    BASE_EXPERIMENT_NAME=${DATASET_NAME}_${DATASET_SIZE}_local_search_k_${K_VALUE}_mem_l_${MEM_L}_mem_k_${MEM_K}_num_worker_threads_${NUM_SEARCH_THREADS}
	    for DIST_SEARCH_MODE in "${DIST_SEARCH_MODE_LIST[@]}"; do
		echo "dist search mode is $DIST_SEARCH_MODE"
		if [[ $DIST_SEARCH_MODE == "SINGLE_SERVER" ]]; then
		    NUM_SERVERS_LIST=(1)
		else
		    NUM_SERVERS_LIST=(2)
		fi 

		# if [[ "$DIST_SEARCH_MODE" == "SCATTER_GATHER_TOP_N" ]]; then
		# 	if [[ ]]; then
		
		# 	    TOP_N_LIST=
		# 	fi
		
		# fi
		
		if [[ "$DIST_SEARCH_MODE" == "DISTRIBUTED_ANN" && ($NUM_ORCHESTRATION_THREADS == 0 || $NUM_SCORING_THREADS == 0) ]]; then
		    echo "Mode is DISTRIBUTEDANN but NUM_ORCHESTRATION_THREADS $NUM_ORCHESTRATION_THREADS and NUM_SCORING_THREADS $NUM_SCORING_THREADS"
		    exit 1
		fi

		if [[ $DIST_SEARCH_MODE == "DISTRIBUTED_ANN" ]]; then
		    BEAM_WIDTH_LIST=(64)
		else
		    BEAM_WIDTH_LIST=(8)
		fi

		for SEND_RATE in ${SEND_RATE_LIST[@]}; do 
		    for NUM_SERVERS in "${NUM_SERVERS_LIST[@]}"; do
			if [[ "$DIST_SEARCH_MODE" == "SCATTER_GATHER_TOP_N" ]]; then
			    if [[ $NUM_SERVERS -eq 5 ]]; then
				TOP_N_LIST=(2 3 4)
			    elif [[ $NUM_SERVERS -eq 10 ]]; then
				TOP_N_LIST=(3 5 7 9)
			    fi
			else
			    TOP_N_LIST=(1)
			fi 
			for TOP_N in ${TOP_N_LIST[@]}; do		    
			    echo "num servers is $NUM_SERVERS"
			    for BEAM_WIDTH in "${BEAM_WIDTH_LIST[@]}"; do
				EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${SEARCH_THREAD_MODE}_${NUM_SERVERS}_server_beam_${BEAM_WIDTH}"
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
					       $MEM_K \
					       $K_VALUE \
					       $TOP_N \
					       $NUM_ORCHESTRATION_THREADS \
					       $NUM_SCORING_THREADS \
					       $SEARCH_THREAD_MODE \
					       $LVEC

				# ${SCRIPT_DIR}/../kill_all_cloudlab_processes.sh
				
			    done
			done
		    done
		done
	    done
	done
    done
done 
