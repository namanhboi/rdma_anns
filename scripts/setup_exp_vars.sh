#!/bin/bash

# This script can be sourced or executed
# - When sourced: exports all variables for use by parent script
# - When executed: prints configuration summary only

# Detect if script is being sourced
SOURCED=0
(return 0 2>/dev/null) && SOURCED=1

# Usage:
# ./setup_exp_vars.sh <num_servers> <dataset_name> <dataset_size> <dist_search_mode> <mode>
# mode: local or distributed

SERVER_STARTING_ADDRESS="10.10.1.1"
BASE_PORT=8000

# --- Argument parsing ---
NUM_SERVERS=$1
DATASET_NAME=$2
DATASET_SIZE=$3
DIST_SEARCH_MODE=$4
MODE=$5



if [ $# -ne 5 ]; then
    echo "Usage: ${BASH_SOURCE[0]} <num_servers> <dataset_name> <dataset_size> <dist_search_mode> <mode>"
    echo "  dataset_name: bigann"
    echo "  dataset_size: 10M or 100M"
    echo "  dist_search_mode: STATE_SEND or SCATTER_GATHER or SINGLE_SERVER"
    echo "  mode: local or distributed"
    [ $SOURCED -eq 1 ] && return 1 || exit 1
fi

# --- Input validation ---
[[ "$DATASET_NAME" != "bigann" ]] && { echo "Error: dataset_name must be 'bigann'"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$DATASET_SIZE" != "10M" && "$DATASET_SIZE" != "100M" ]] && { echo "Error: dataset_size must be 10M or 100M"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$DIST_SEARCH_MODE" != "STATE_SEND" && "$DIST_SEARCH_MODE" != "SCATTER_GATHER" && "$DIST_SEARCH_MODE" != "SINGLE_SERVER" ]] && { echo "Error: dist_search_mode must be STATE_SEND or SCATTER_GATHER or SINGLE_SERVER"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$MODE" != "local" && "$MODE" != "distributed" ]] && { echo "Error: mode must be local or distributed"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }

# Numeric validation
[[ ! "$NUM_SERVERS" =~ ^[0-9]+$ ]] && { echo "Error: num_servers must be a positive integer"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }
[[ "$NUM_SERVERS" -lt 1 ]] && { echo "Error: num_servers must be at least 1"; [ $SOURCED -eq 1 ] && return 1 || exit 1; }

# --- Dataset metadata ---
DATA_TYPE="uint8"
DIMENSION=128
METRIC="l2"

# --- Mode-based prefix path ---
if [ "$MODE" == "local" ]; then
    ANNGRAHPS_PREFIX="$HOME/big-ann-benchmarks/data"
else
    ANNGRAHPS_PREFIX="/mydata/local/anngraphs"
fi

# --- Graph prefix path ---
if [[ "$DIST_SEARCH_MODE" == "SINGLE_SERVER" ]]; then
    GRAPH_PREFIX="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/pipeann_${DATASET_SIZE}"
else
    if [ "$DIST_SEARCH_MODE" == "STATE_SEND" ]; then
	PREFIX="global_partitions"
	GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_partition"
    else
	PREFIX="clusters"
	GRAPH_SUFFIX="pipeann_${DATASET_SIZE}_cluster"
    fi
    GRAPH_PREFIX="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/${PREFIX}_${NUM_SERVERS}/${GRAPH_SUFFIX}"
fi

# --- Query and truthset paths ---
QUERY_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/query.public.10K.u8bin"
TRUTHSET_BIN="${ANNGRAHPS_PREFIX}/${DATASET_NAME}/${DATASET_SIZE}/bigann-${DATASET_SIZE}"

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
    for ((i=0; i<=NUM_SERVERS; i++)); do
        PEER_IPS+=("$OCT1.$OCT2.$OCT3.$((OCT4+i)):$BASE_PORT")
    done
fi

# --- CloudLab external hostnames for SSH from laptop ---
if [[ "$MODE" == "distributed" ]]; then
    # Full list of available CloudLab hosts
    ALL_CLOUDLAB_HOSTS=(
        "namanh@er088.utah.cloudlab.us"
        "namanh@er084.utah.cloudlab.us"
        "namanh@er075.utah.cloudlab.us"
        "namanh@er068.utah.cloudlab.us"
        "namanh@er102.utah.cloudlab.us"
        "namanh@er118.utah.cloudlab.us"
    )
    
    # Only take NUM_SERVERS + 1 hosts (servers + client)
    NEEDED_HOSTS=$((NUM_SERVERS + 1))
    
    if [ $NEEDED_HOSTS -gt ${#ALL_CLOUDLAB_HOSTS[@]} ]; then
        echo "Error: Need $NEEDED_HOSTS CloudLab hosts but only ${#ALL_CLOUDLAB_HOSTS[@]} available"
        [ $SOURCED -eq 1 ] && return 1 || exit 1
    fi
    
    CLOUDLAB_HOSTS=("${ALL_CLOUDLAB_HOSTS[@]:0:$NEEDED_HOSTS}")
fi


# --- Server parameters ---
NUM_SEARCH_THREADS=8
USE_MEM_INDEX=true
NUM_QUERIES_BALANCE=8
USE_BATCHING=true
MAX_BATCH_SIZE=1
USE_COUNTER_THREAD=true
USE_LOGGING=true
COUNTER_SLEEP_MS=10
# --- Client parameters ---
NUM_CLIENT_THREADS=1
# 10 15 20 25 30 35 40 50 60 80 120 200 400
LVEC="10 15 20 25 30 35 40 50 60 80 120 200 400"
BEAM_WIDTH=1
K_VALUE=10
MEM_L=10
RECORD_STATS=true
SEND_RATE=0

EXPERIMENT_NAME=${DIST_SEARCH_MODE}_${MODE}_${DATASET_NAME}_${DATASET_SIZE}_${NUM_SERVERS}_${COUNTER_SLEEP_MS}_MS_MAX_BATCH_SIZE_${MAX_BATCH_SIZE}_K_${K_VALUE}_LVEC_${LVEC// /_}
# --- Export variables ---


export NUM_SEARCH_THREADS USE_MEM_INDEX NUM_QUERIES_BALANCE USE_BATCHING MAX_BATCH_SIZE USE_COUNTER_THREAD COUNTER_SLEEP_MS
export NUM_CLIENT_THREADS LVEC BEAM_WIDTH K_VALUE MEM_L RECORD_STATS SEND_RATE
export NUM_SERVERS DATASET_NAME DATASET_SIZE DATA_TYPE DIMENSION METRIC DIST_SEARCH_MODE MODE
export ANNGRAHPS_PREFIX GRAPH_PREFIX QUERY_BIN TRUTHSET_BIN
export PEER_IPS
export PEER_IPS_STR="${PEER_IPS[*]}"
export USER
export EXPERIMENT_NAME
export USE_LOGGING

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
