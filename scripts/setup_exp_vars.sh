#!/usr/bin/bash

# This script can be sourced or executed
# - When sourced: exports all variables for use by parent script
# - When executed: prints configuration summary only

# Detect if script is being sourced
SOURCED=0
(return 0 2>/dev/null) && SOURCED=1


SERVER_STARTING_ADDRESS="10.10.1.1"
BASE_PORT=8000


if [ $# -lt 19 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <master_log_folder_name> <num_servers> <dataset_name> <dataset_size> <dist_search_mode> <mode> <num_search_thread> <max_batch_size> <overlap>"
    echo "  master_log_folder_name: example : testing"
    echo "  dataset_name: bigann"
    echo "  dataset_size: 10M or 100M or 1B"
    echo "  dist_search_mode: STATE_SEND or SCATTER_GATHER or SINGLE_SERVER or DISTRIBUTED_ANN"
    echo "  mode: local or distributed"
    echo "  num_search_thread: number"
    echo "  max_batch_size: number"
    echo "  overlap: true or false"
    echo "  beamwidth : "
    echo "  num_client_threads : 1 for anything except Distributedann, for distributedann, this is the number of ochestration threads"
    echo "  use_counter_thread : use the counter thread or not. Right now counter thread is not yet implemented for distributedann"
    echo "  use_logging : logging is to get the message sizes in the handler and the serialization time rn. Need to remove the serialization time stuff"
    echo "  send_rate : this is the number of queries you want to send per second. "
    echo "  write_query_csv : whether or not to record information about each individual query into a csv file for each L "
    echo "  num_queries_to_send : num queries to send, default to a large number"
    
    [ $SOURCED -eq 1 ] && return 1 || exit 1
fi


MASTER_LOG_FOLDER_NAME=$1
NUM_SERVERS=$2
DATASET_NAME=$3
DATASET_SIZE=$4
DIST_SEARCH_MODE=$5
MODE=$6
NUM_SEARCH_THREADS=$7
MAX_BATCH_SIZE=$8
OVERLAP=$9
BEAM_WIDTH=${10}
NUM_CLIENT_THREADS=${11}
USE_COUNTER_THREAD=${12}
USE_LOGGING=${13}
SEND_RATE=${14}
WRITE_QUERY_CSV=${15}
NUM_QUERIES_TO_SEND=${16}
MEM_L=${17}
K_VALUE=${18}
TOP_N=${19}
shift 19
LVEC=$(printf " %s" "$@")
LVEC=${LVEC:1}

# --- Input validation ---
[[ "$DATASET_NAME" != "bigann" && "$DATASET_NAME" != "deep1b" && "$DATASET_NAME" != "MSSPACEV1B" && "$DATASET_NAME" != "text2image1B" && "$DATASET_NAME" != "OpenAIArXiv" ]] && { echo "Error: dataset_name must be 'bigann', deep1b, 'MSSPACEV1B', 'text2image1B', 'OpenAIArxiv'"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
# [[ "$DATASET_SIZE" != "10M" && "$DATASET_SIZE" != "100M" && "$DATASET_SIZE" != "1B" ]] && { echo "Error: dataset_size must be 10M or 100M or 1B"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$DIST_SEARCH_MODE" != "STATE_SEND" && "$DIST_SEARCH_MODE" != "SCATTER_GATHER" && "$DIST_SEARCH_MODE" != "SINGLE_SERVER" && "$DIST_SEARCH_MODE" != "DISTRIBUTED_ANN" && "$DIST_SEARCH_MODE" != "STATE_SEND_CLIENT_GATHER" && "$DIST_SEARCH_MODE" != "SCATTER_GATHER_TOP_N"  ]]  && { echo "Error: dist_search_mode must be STATE_SEND or SCATTER_GATHER or SINGLE_SERVER"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$MODE" != "local" && "$MODE" != "distributed" ]] && { echo "Error: mode must be local or distributed"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }

# Numeric validation
[[ ! "$NUM_SERVERS" =~ ^[0-9]+$ ]] && { echo "Error: num_servers must be a positive integer"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$NUM_SERVERS" -lt 1 ]] && { echo "Error: num_servers must be at least 1"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }


USE_MEM_INDEX=false
if [[ $MEM_L != "0" ]]; then
    USE_MEM_INDEX=true
fi


# --- Mode-based prefix path ---
if [ "$MODE" == "local" ]; then
    ANNGRAHPS_PREFIX="$HOME/big-ann-benchmarks/data"
else
    ANNGRAHPS_PREFIX="/mydata/local/anngraphs"
fi

# --- Dataset metadata ---

if [[ "$DATASET_NAME" == "bigann" ]]; then
    DATA_TYPE="uint8"
    DIMENSION=128
    METRIC="l2"

    QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/query.public.10K.u8bin"
    if [[ "${DATASET_SIZE}" == "1B" ]]; then
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/GT.public.1B.ibin"
    else
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/bigann-${DATASET_SIZE}"
    fi
elif [[ "$DATASET_NAME" == "deep1b" ]]; then
    DATA_TYPE="float"
    DIMENSION=96
    METRIC="l2"
    QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/query.public.10K.fbin"
    TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/deep-${DATASET_SIZE}"
elif [[ "$DATASET_NAME" == "MSSPACEV1B" ]]; then
    DATA_TYPE="int8"
    DIMENSION=100
    METRIC="l2"
    QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/query.i8bin"
    if [[ "${DATASET_SIZE}" == "1B" ]]; then
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/public_query_gt100.bin"
    else
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/msspacev-gt-100M"	
    fi
elif [[ "$DATASET_NAME" == "text2image1B" ]]; then
    DATA_TYPE="float"
    DIMENSION=200
    METRIC="mips"
    if [[ "$MODE" == "local" ]]; then
	QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/query.heldout.30K.fbin"
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/diskann_recomputed_gt100-heldout.30K.fbin"
    else
	echo "ERROR: text2image only supported for local currently"
	exit 1
    fi
elif [[ "$DATASET_NAME" == "OpenAIArXiv" ]]; then
    DATA_TYPE="float"
    DIMENSION=1536
    METRIC="cosine"
    if [[ "$MODE" == "local" ]]; then
	QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/openai_query.bin"
	TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/openai-100K"
    else
	echo "ERROR: OpenAIArXiv only supports local rn"
	exit 1
    fi
fi



# --- Graph prefix path ---
if [[ "$DIST_SEARCH_MODE" == "SINGLE_SERVER" ]]; then
    GRAPH_PREFIX="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/pipeann_${DATASET_SIZE}"
    PREFIX=""
    GRAPH_SUFFIX=""
else
    if [[ $OVERLAP == "true" ]]; then
	if [ "$DIST_SEARCH_MODE" == "STATE_SEND" || "$DIST_SEARCH_MODE" == "STATE_SEND_CLIENT_GATHER" ]; then
	    PREFIX="global_overlap_partitions"
	    GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_partition"
	else
	    PREFIX="overlap_clusters"
	    GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_cluster"
	fi	
    else
	if [[ ("$DIST_SEARCH_MODE" == "STATE_SEND_CLIENT_GATHER") || ("$DIST_SEARCH_MODE" == "STATE_SEND") || ("$DIST_SEARCH_MODE" == "DISTRIBUTED_ANN") ]]; then
	    PREFIX="global_partitions"
	    GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_partition"
	else
	    PREFIX="clusters"
	    GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_cluster"
	fi
    fi
    GRAPH_PREFIX="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/${PREFIX}_${NUM_SERVERS}/${GRAPH_SUFFIX}"
fi

# --- Query and truthset paths ---

MEDOID_FILE="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/medoids.bin"
DISTRIBUTEDANN_CLIENT_PARTITION_ASSIGNMENT_FILE="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/${PREFIX}_${NUM_SERVERS}/${GRAPH_SUFFIX}_assignment.bin"


# --- User configuration ---
USER_LOCAL=nam
USER_REMOTE=namanh
if [[ "$MODE" == "local" ]]; then
    USER=$USER_LOCAL
else
    USER=$USER_REMOTE
fi

# --- Generate peer IPs (servers + client) ---
PEER_IPS=()

if [ "$MODE" == "local" ]; then
    for ((i=0; i<=NUM_SERVERS; i++)); do
        PEER_IPS+=("127.0.0.1:$((BASE_PORT+i))")
    done
else
    IFS='.' read -r OCT1 OCT2 OCT3 OCT4 <<< "$SERVER_STARTING_ADDRESS"
    LAST_IP=$((OCT4 + NUM_SERVERS))
    if [ $LAST_IP -gt 255 ]; then
        echo "Error: IP range overflow (needs $((NUM_SERVERS+1)) IPs starting at $SERVER_STARTING_ADDRESS)"
        [ $SOURCED -eq 1 ] && return 1 || exit 1
    fi
    for ((i=0; i<NUM_SERVERS; i++)); do
        PEER_IPS+=("$OCT1.$OCT2.$OCT3.$((OCT4+i)):$BASE_PORT")
    done
    PEER_IPS+=("$OCT1.$OCT2.$OCT3.$((OCT4+NUM_SERVERS-1)):$((BASE_PORT+1))")
fi

# --- CloudLab external hostnames for SSH from laptop ---
if [[ "$MODE" == "distributed" ]]; then
    # Full list of available CloudLab hosts
    ALL_CLOUDLAB_HOSTS=(
        "namanh@er039.utah.cloudlab.us"
	"namanh@er082.utah.cloudlab.us"
	"namanh@er076.utah.cloudlab.us"
	"namanh@er050.utah.cloudlab.us"
	"namanh@er104.utah.cloudlab.us" 
	"namanh@er124.utah.cloudlab.us"
	"namanh@er001.utah.cloudlab.us"
	"namanh@er040.utah.cloudlab.us"
	"namanh@er032.utah.cloudlab.us"
	"namanh@er126.utah.cloudlab.us"					
    )
    
    # Only take NUM_SERVERS + 1 hosts (servers + client)
    NEEDED_HOSTS=$((NUM_SERVERS))
    
    if [ $NEEDED_HOSTS -gt ${#ALL_CLOUDLAB_HOSTS[@]} ]; then
        echo "Error: Need $NEEDED_HOSTS CloudLab hosts but only ${#ALL_CLOUDLAB_HOSTS[@]} available"
        [ $SOURCED -eq 1 ] && return 1 || exit 1
    fi
    
    CLOUDLAB_HOSTS=("${ALL_CLOUDLAB_HOSTS[@]:0:$NEEDED_HOSTS}")
fi


# --- Server parameters ---


NUM_QUERIES_BALANCE=8
USE_BATCHING=true


COUNTER_SLEEP_MS=100
# --- Client parameters ---
# 10 15 20 25 30 35 40 50 60 80 120 200 400
# LVEC="10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 45 50 55 60 65 70 80 90 100 120 140 160 180 200 225 250 275 300 375"
# LVEC="10 15 20 25 30 35 40 50 60 80 120 200 400"
# LVEC="400"
# LVEC="65 70 80 100 120 140 160"
# LVEC="400"
K_VALUE=10
RECORD_STATS=true


EXPERIMENT_NAME=${DIST_SEARCH_MODE}_${MODE}_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${COUNTER_SLEEP_MS}_MS_NUM_SEARCH_THREADS_${NUM_SEARCH_THREADS}_MAX_BATCH_SIZE_${MAX_BATCH_SIZE}_K_${K_VALUE}_OVERLAP_${OVERLAP}_BEAMWIDTH_${BEAM_WIDTH}
# --- Export variables ---


export NUM_SEARCH_THREADS USE_MEM_INDEX NUM_QUERIES_BALANCE USE_BATCHING MAX_BATCH_SIZE USE_COUNTER_THREAD COUNTER_SLEEP_MS 
export NUM_CLIENT_THREADS LVEC BEAM_WIDTH K_VALUE MEM_L RECORD_STATS SEND_RATE WRITE_QUERY_CSV NUM_QUERIES_TO_SEND TOP_N MEDOID_FILE
export NUM_SERVERS DATASET_NAME DATASET_SIZE DATA_TYPE DIMENSION METRIC DIST_SEARCH_MODE MODE
export ANNGRAHPS_PREFIX GRAPH_PREFIX QUERY_BIN TRUTHSET_BIN
export PEER_IPS
export PEER_IPS_STR="${PEER_IPS[*]}"
export USER
export EXPERIMENT_NAME
export USE_LOGGING
export MASTER_LOG_FOLDER_NAME
export DISTRIBUTEDANN_CLIENT_PARTITION_ASSIGNMENT_FILE

if [[ "$MODE" == "distributed" ]]; then
    export CLOUDLAB_HOSTS
    export CLOUDLAB_HOSTS_STR="${CLOUDLAB_HOSTS[*]}"
fi

# --- Output summary (only if executed directly) ---
if [ $SOURCED -eq 0 ]; then
    echo "========================================"
    echo " Configuration Summary"
    echo "========================================"
    echo "Mode:                $MODE"
    echo "Servers:             $NUM_SERVERS"
    echo "Dataset:             $DATASET_NAME"
    echo "Dataset size:        $DATASET_SIZE"
    echo "Data type:           $DATA_TYPE"
    echo "Dimension:           $DIMENSION"
    echo "Metric:              $METRIC"
    echo "Dist search mode:    $DIST_SEARCH_MODE"
    echo "Graph prefix path:   $GRAPH_PREFIX"
    echo "Query binary:        $QUERY_BIN"
    echo "Truthset binary:     $TRUTHSET_BIN"
    echo "WRITE_QUERY_CSV:     $WRITE_QUERY_CSV"    
    echo
    echo "Peer IPs (servers + client):"
    for ip in "${PEER_IPS[@]}"; do
        echo "  $ip"
    done
    if [[ "$MODE" == "distributed" ]]; then
        echo
        echo "CloudLab SSH Hosts:"
        for host in "${CLOUDLAB_HOSTS[@]}"; do
            echo "  $host"
        done
    fi
    echo "========================================"
fi
