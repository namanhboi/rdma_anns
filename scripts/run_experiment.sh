#!/bin/bash

set -euo pipefail

# Disable SSH agent to prevent interference
unset SSH_AUTH_SOCK
unset SSH_AGENT_PID

# --- Configuration ---
echo "Loading configuration..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Now source relative to script location
source "${SCRIPT_DIR}/setup_exp_vars.sh" $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}

# --- Helper Functions ---
WORKDIR="$HOME/workspace/rdma_anns/"
if [[ "$MODE" == "distributed" ]]; then 
    WORKDIR="/users/$USER/workspace/rdma_anns/"
fi
echo ${WORKDIR}

# SSH options to suppress agent messages
SSH_OPTS="-o StrictHostKeyChecking=no -o ForwardAgent=no -o LogLevel=ERROR"

# Reconstruct arrays from exported strings
IFS=' ' read -ra PEER_IPS <<< "$PEER_IPS_STR"

if [[ "$MODE" == "distributed" ]]; then
    IFS=' ' read -ra CLOUDLAB_HOSTS <<< "$CLOUDLAB_HOSTS_STR"
fi

sshCommandAsync() {
    local server="$1"
    local command="$2"
    local outfile="${3:-}"
    
    local output=$(ssh $SSH_OPTS "$server" /bin/bash <<EOF
nohup /bin/bash -c '$command' > '$outfile' 2>&1 &
echo \$!
EOF
    )
    
    # Extract only the PID (last number in output)
    echo "$output" | grep -o '[0-9]\+' | tail -1
}

sshCommandSync() {
    local server="$1"
    local command="$2"
    local outfile="${3:-}"
    
    ssh $SSH_OPTS "$server" /bin/bash <<EOF
$command
EOF
}

sshStopCommand() {
    local server="$1"
    local pid="$2"
    
    # Validate PID is numeric
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        ssh $SSH_OPTS "$server" /bin/bash <<EOF
kill -2 $pid 2>/dev/null || true
EOF
    else
        echo "Warning: Invalid PID '$pid' for server $server" >&2
    fi
}



# Client is always the last peer
CLIENT_ID=$((${#PEER_IPS[@]} - 1))
CLIENT_IP="${PEER_IPS[$CLIENT_ID]}"

echo "========================================"
echo " Launching Distributed ANN System"
echo "========================================"
echo "Configuration loaded:"
echo "  Mode: $MODE"
echo "  Servers: $NUM_SERVERS"
echo "  Client ID: $CLIENT_ID"
echo "  Dataset: $DATASET_NAME ($DATASET_SIZE)"
echo "  Dist search mode: $DIST_SEARCH_MODE"
echo "  Working directory: $WORKDIR"
echo "  Graph prefix: $GRAPH_PREFIX"
echo "  Query file: $QUERY_BIN"
echo "  Ground truth: $TRUTHSET_BIN"
echo

# Create log directory

# Build address list string (tcp:// prefixed, space-separated)
ADDRESS_LIST_STR=""
for ip in "${PEER_IPS[@]}"; do
  ADDRESS_LIST_STR+="tcp://${ip} "
done
ADDRESS_LIST_STR=$(echo $ADDRESS_LIST_STR | sed 's/ $//')  # Remove trailing space

# --- Extract unique hosts and create remote directories ---
echo "========================================"
echo " Preparing remote hosts"
echo "========================================"


LOG_DIRNAME="logs_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
LOCAL_LOG_DIR="$HOME/workspace/rdma_anns/logs/${MASTER_LOG_FOLDER_NAME}/${LOG_DIRNAME}"
mkdir -p "$LOCAL_LOG_DIR"
REMOTE_LOG_DIR="${WORKDIR}/logs/${MASTER_LOG_FOLDER_NAME}/${LOG_DIRNAME}"

# For distributed mode, use CloudLab hostnames
echo "Creating log directories on CloudLab hosts..."
for CLOUDLAB_HOST in "${CLOUDLAB_HOSTS[@]}"; do
    echo "  Creating directories on $CLOUDLAB_HOST..."
    sshCommandSync "$CLOUDLAB_HOST" "mkdir -p ${REMOTE_LOG_DIR}"
    echo "    ✓ Ready: $CLOUDLAB_HOST"
done

echo

# --- Start Servers ---
echo "========================================"
echo " Starting servers"
echo "========================================"

declare -A SERVER_PIDS
declare -A SERVER_HOSTS

for i in $(seq 0 $((NUM_SERVERS - 1))); do
  SERVER_IP="${PEER_IPS[$i]}"
  
  # Determine which host to SSH to
  if [[ "$MODE" == "local" ]]; then
      SSH_HOST="$USER@127.0.0.1"
  else
      SSH_HOST="${CLOUDLAB_HOSTS[$i]}"
  fi
  
  echo "  Server $i via $SSH_HOST (internal: tcp://$SERVER_IP)"
  COUNTER_CSV=${REMOTE_LOG_DIR}/counter_${i}.csv
  LOG_FILE=${REMOTE_LOG_DIR}/log_${i}.txt
  # Build server command with all arguments
  SERVER_CMD="$WORKDIR/build/src/state_send/state_send_server \
    --server_peer_id=$i \
    --address_list $ADDRESS_LIST_STR \
    --data_type=$DATA_TYPE \
    --index_path_prefix=${GRAPH_PREFIX} \
    --num_search_threads=$NUM_SEARCH_THREADS \
    --use_mem_index=$USE_MEM_INDEX \
    --metric=$METRIC \
    --num_queries_balance=$NUM_QUERIES_BALANCE \
    --dist_search_mode=$DIST_SEARCH_MODE \
    --use_batching=$USE_BATCHING \
    --max_batch_size=$MAX_BATCH_SIZE \
    --use_counter_thread=$USE_COUNTER_THREAD \
    --counter_csv=$COUNTER_CSV \
    --counter_sleep_ms=$COUNTER_SLEEP_MS \
    --use_logging=$USE_LOGGING \
    --log_file=$LOG_FILE"
  echo ${SERVER_CMD}
  
  # Launch server via SSH
  REMOTE_PID=$(sshCommandAsync "$SSH_HOST" \
    "cd ${WORKDIR} && $SERVER_CMD" \
    "${REMOTE_LOG_DIR}/server_${i}.log")
  
  SERVER_PIDS[$i]=$REMOTE_PID
  SERVER_HOSTS[$i]=$SSH_HOST
  echo "    PID: $REMOTE_PID"
  echo
done

echo "Waiting for servers to initialize..."
if [[ "$MODE" == "local" ]]; then 
    sleep 30
else
    if [[ "$DATASET_SIZE" == "1B" ]]; then
	sleep 120
    else
	sleep 30
    fi
fi


# --- Start Client ---
echo "========================================"
echo " Starting client"
echo "========================================"

# Determine which host to SSH to for client
if [[ "$MODE" == "local" ]]; then
    CLIENT_SSH_HOST="$USER@127.0.0.1"
else
    CLIENT_SSH_HOST="${CLOUDLAB_HOSTS[$((CLIENT_ID-1))]}"
fi

echo "  Client via $CLIENT_SSH_HOST"
echo "  Client peer ID: $CLIENT_ID"
echo "  Client address: tcp://$CLIENT_IP"



# Build client command with all arguments
CLIENT_CMD="$WORKDIR/build/benchmark/state_send/run_benchmark_state_send_tcp \
  --num_client_thread=$NUM_CLIENT_THREADS \
  --dim=$DIMENSION \
  --query_bin=$QUERY_BIN \
  --truthset_bin=$TRUTHSET_BIN \
  --L $LVEC \
  --beam_width=$BEAM_WIDTH \
  --K=$K_VALUE \
  --mem_L=$MEM_L \
  --record_stats=$RECORD_STATS \
  --dist_search_mode=$DIST_SEARCH_MODE \
  --client_peer_id=$CLIENT_ID \
  --send_rate=$SEND_RATE \
  --address_list $ADDRESS_LIST_STR \
  --data_type=$DATA_TYPE \
  --result_output_folder=$REMOTE_LOG_DIR \
  --partition_assignment_file=${DISTRIBUTEDANN_CLIENT_PARTITION_ASSIGNMENT_FILE} \
  --write_query_csv=${WRITE_QUERY_CSV}"

echo ${CLIENT_CMD}
# Launch client via SSH
CLIENT_REMOTE_PID=$(sshCommandAsync "$CLIENT_SSH_HOST" \
  "cd ${WORKDIR} && $CLIENT_CMD" \
  "${REMOTE_LOG_DIR}/client.log")

echo "  PID: $CLIENT_REMOTE_PID"

echo
echo "========================================"
echo " System Running ($MODE mode)"
echo "========================================"
echo "Server PIDs:"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  echo "  Server $i via ${SERVER_HOSTS[$i]}: PID ${SERVER_PIDS[$i]}"
done
echo
echo "Client via $CLIENT_SSH_HOST: PID $CLIENT_REMOTE_PID"
echo
echo "Logs available at: ${REMOTE_LOG_DIR}/"
echo "  Server logs: ${REMOTE_LOG_DIR}/server_*.log"
echo "  Client log: ${REMOTE_LOG_DIR}/client.log"
echo "========================================"

# --- Cleanup handler (for Ctrl+C) ---
cleanup() {
  echo
  echo "Interrupted! Shutting down gracefully..."
  
  if [[ "$MODE" == "local" ]]; then
      # Kill everything on localhost
      echo "  Stopping all processes on localhost..."
      ssh $SSH_OPTS "$USER@127.0.0.1" /bin/bash <<'EOF' || true
# Send SIGINT to all processes
pkill -2 -f 'state_send_server' 2>/dev/null
pkill -2 -f 'run_benchmark_state_send_tcp' 2>/dev/null
sleep 1
# If still running, force kill 
pkill -9 -f 'state_send_server' 2>/dev/null
pkill -9 -f 'run_benchmark_state_send_tcp' 2>/dev/null
EOF
  else
      # Kill everything on all CloudLab hosts
      for CLOUDLAB_HOST in "${CLOUDLAB_HOSTS[@]}"; do
          echo "  Stopping all processes on $CLOUDLAB_HOST..."
          ssh $SSH_OPTS "$CLOUDLAB_HOST" /bin/bash <<'EOF' || true
# Send SIGINT to all processes
pkill -2 -f 'state_send_server' 2>/dev/null
pkill -2 -f 'run_benchmark_state_send_tcp' 2>/dev/null
sleep 1
# If still running, force kill
pkill -9 -f 'state_send_server' 2>/dev/null
pkill -9 -f 'run_benchmark_state_send_tcp' 2>/dev/null
EOF
      done
  fi
  
  echo "All processes stopped."
  exit 0
}
trap cleanup SIGINT SIGTERM

# --- Wait for client to finish ---
echo "Waiting for client to complete..."

# Poll the client process until it exits
while ssh $SSH_OPTS "$CLIENT_SSH_HOST" "kill -0 $CLIENT_REMOTE_PID" 2>/dev/null; do
  sleep 2
done

echo
echo "Client finished!"
echo "Stopping servers gracefully (sending SIGINT)..."

# Send SIGINT to all servers
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  SSH_HOST="${SERVER_HOSTS[$i]}"
  PID="${SERVER_PIDS[$i]}"
  echo "  Sending SIGINT to server $i via $SSH_HOST (PID: $PID)..."
  sshStopCommand "$SSH_HOST" "$PID" || true
done

echo "Waiting for servers to exit gracefully..."
sleep 5

echo "All processes stopped successfully!"

# # --- Copy logs back to local machine ---
# echo
# echo "========================================"
# echo " Organizing logs"
# echo "========================================"

# # Create local log directory with timestamp

# if [[ "$MODE" == "distributed" ]]; then

#     echo "Local log directory: $LOCAL_LOG_DIR"
#     echo

#     echo "Running in DISTRIBUTED mode - copying logs from remote hosts..."
    
#     # Copy logs from each CloudLab host
#     for i in "${!CLOUDLAB_HOSTS[@]}"; do
#         CLOUDLAB_HOST="${CLOUDLAB_HOSTS[$i]}"
#         echo "  Copying logs from $CLOUDLAB_HOST..."
        
#         # Use tar over SSH
#         ssh $SSH_OPTS "$CLOUDLAB_HOST" "cd ${REMOTE_LOG_DIR} 2>/dev/null && tar cf - . 2>/dev/null" | tar xf - -C "$LOCAL_LOG_DIR/" 2>/dev/null && {
#             echo "    ✓ Logs saved to: $LOCAL_LOG_DIR"
#         } || {
#             echo "    ⚠ Could not copy logs from $CLOUDLAB_HOST"
#         }
#     done
# fi

# echo
# echo "All logs organized successfully!"
# echo "Logs location: $LOCAL_LOG_DIR"
# echo
# echo "Done!"



# --- Copy logs back to local machine ---
echo
echo "========================================"
echo " Organizing logs"
echo "========================================"

if [[ "$MODE" == "distributed" ]]; then

    echo "Local log directory: $LOCAL_LOG_DIR"
    echo

    echo "Running in DISTRIBUTED mode - copying client.log from last CloudLab host..."
    
    # Get the last CloudLab host (where the client runs)
    LAST_HOST_INDEX=$((${#CLOUDLAB_HOSTS[@]} - 1))
    LAST_CLOUDLAB_HOST="${CLOUDLAB_HOSTS[$LAST_HOST_INDEX]}"
    
    echo "  Copying client.log from $LAST_CLOUDLAB_HOST..."
    
    # Copy only client.log from the last host
    scp $SSH_OPTS "$LAST_CLOUDLAB_HOST:${REMOTE_LOG_DIR}/client.log" "$LOCAL_LOG_DIR/client.log" && {
        echo "    ✓ client.log saved to: $LOCAL_LOG_DIR/client.log"
    } || {
        echo "    ⚠ Could not copy client.log from $LAST_CLOUDLAB_HOST"
    }
fi

echo
echo "Client log saved successfully!"
echo "Log location: $LOCAL_LOG_DIR/client.log"
echo
echo "Done!"


### making the figuers automatically
# mkdir "${LOCAL_LOG_DIR}/figures"
# python3.10 "$HOME/workspace/rdma_anns/scripts/plot_counter_data.py" -i ${LOCAL_LOG_DIR}  -o "${LOCAL_LOG_DIR}/figures"
# python3.10 "$HOME/workspace/rdma_anns/scripts/plot_query_data.py" -i ${LOCAL_LOG_DIR} -o "${LOCAL_LOG_DIR}/figures"
# python3.10 "$HOME/workspace/rdma_anns/scripts/analyze_log.py" ${LOCAL_LOG_DIR} "${LOCAL_LOG_DIR}/figures"

