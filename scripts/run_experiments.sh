#!/bin/bash

set -euo pipefail

# --- Configuration ---
echo "Loading configuration..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Now source relative to script location
source "${SCRIPT_DIR}/setup_exp_vars.sh" $1 $2 $3 $4 $5



# --- Helper Functions ---
WORKDIR="$HOME/workspace/rdma_anns/"
echo ${WORKDIR}

# SSH options to suppress agent messages
SSH_OPTS="-o StrictHostKeyChecking=no -o ForwardAgent=no -o LogLevel=ERROR"

sshCommandAsync() {
    local server="$1"
    local command="$2"
    local outfile="${3:-}"
    
    local output=$(ssh $SSH_OPTS "$USER@$server" /bin/bash <<EOF
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
    
    ssh $SSH_OPTS "$USER@$server" /bin/bash <<EOF
$command
EOF
}

sshStopCommand() {
    local server="$1"
    local pid="$2"
    
    # Validate PID is numeric
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        ssh $SSH_OPTS "$USER@$server" /bin/bash <<EOF
kill -2 $pid 2>/dev/null || true
EOF
    else
        echo "Warning: Invalid PID '$pid' for server $server" >&2
    fi
}

# Reconstruct PEER_IPS array from exported string
IFS=' ' read -ra PEER_IPS <<< "$PEER_IPS_STR"

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
echo "  Use tags: $USE_TAGS"
echo "  Enable locs: $ENABLE_LOCS"
echo "  Working directory: $WORKDIR"
echo "  Graph prefix: $GRAPH_PREFIX"
echo "  Query file: $QUERY_BIN"
echo "  Ground truth: $TRUTHSET_BIN"
echo

# Create log directory
mkdir -p logs

# --- Server parameters ---
NUM_SEARCH_THREADS=8
USE_MEM_INDEX=true
NUM_QUERIES_BALANCE=8
USE_BATCHING=true
MAX_BATCH_SIZE=16

# --- Client parameters ---
NUM_CLIENT_THREADS=1
LVEC="10 15 20 25 30 35 40 50 60 80 120 200 400"
BEAM_WIDTH=1
K_VALUE=10
MEM_L=10
RECORD_STATS=true
SEND_RATE=0

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

declare -A HOSTS
for ip in "${PEER_IPS[@]}"; do
  HOST=$(echo $ip | cut -d: -f1)
  HOSTS[$HOST]=1
done

echo "Creating log directories on hosts..."
for HOST in "${!HOSTS[@]}"; do
  echo "  Creating directories on $HOST..."
  sshCommandSync "$HOST" "mkdir -p ${WORKDIR}/logs"
  echo "    ✓ Ready: $HOST"
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
  HOST=$(echo $SERVER_IP | cut -d: -f1)
  
  echo "  Server $i on $HOST (tcp://$SERVER_IP)"
  
  # Build server command with all arguments
  SERVER_CMD="$WORKDIR/build/src/state_send/state_send_server \
    --server_peer_id=$i \
    --address_list $ADDRESS_LIST_STR \
    --data_type=$DATA_TYPE \
    --index_path_prefix=${GRAPH_PREFIX} \
    --num_search_threads=$NUM_SEARCH_THREADS \
    --use_tags=$USE_TAGS \
    --enable_locs=$ENABLE_LOCS \
    --use_mem_index=$USE_MEM_INDEX \
    --metric=$METRIC \
    --num_queries_balance=$NUM_QUERIES_BALANCE \
    --dist_search_mode=$DIST_SEARCH_MODE \
    --use_batching=$USE_BATCHING \
    --max_batch_size=$MAX_BATCH_SIZE"
  echo ${SERVER_CMD}
  
  # Launch server via SSH
  REMOTE_PID=$(sshCommandAsync "$HOST" \
    "cd ${WORKDIR} && $SERVER_CMD" \
    "${WORKDIR}/logs/server_${i}.log")
  
  SERVER_PIDS[$i]=$REMOTE_PID
  SERVER_HOSTS[$i]=$HOST
  echo "    PID: $REMOTE_PID"
  echo
done

echo "Waiting for servers to initialize..."
sleep 5

# --- Start Client ---
echo "========================================"
echo " Starting client"
echo "========================================"

CLIENT_HOST=$(echo $CLIENT_IP | cut -d: -f1)

echo "  Client on $CLIENT_HOST"
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
  --data_type=$DATA_TYPE"

echo ${CLIENT_CMD}
# Launch client via SSH
CLIENT_REMOTE_PID=$(sshCommandAsync "$CLIENT_HOST" \
  "cd ${WORKDIR} && $CLIENT_CMD" \
  "${WORKDIR}/logs/client.log")

echo "  PID: $CLIENT_REMOTE_PID"

echo
echo "========================================"
echo " System Running ($MODE mode)"
echo "========================================"
echo "Server PIDs:"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  echo "  Server $i on ${SERVER_HOSTS[$i]}: PID ${SERVER_PIDS[$i]}"
done
echo
echo "Client on $CLIENT_HOST: PID $CLIENT_REMOTE_PID"
echo
echo "Logs available at: ${WORKDIR}/logs/"
echo "  Server logs: logs/server_*.log"
echo "  Client log: logs/client.log"
echo "========================================"

# --- Cleanup handler (for Ctrl+C) ---
cleanup() {
  echo
  echo "Interrupted! Shutting down gracefully..."
  
  # Stop client first
  echo "  Stopping client on $CLIENT_HOST (PID: $CLIENT_REMOTE_PID)..."
  sshStopCommand "$CLIENT_HOST" "$CLIENT_REMOTE_PID" 2>/dev/null || true
  sleep 2
  
  # Then stop servers
  echo "  Stopping servers..."
  for i in $(seq 0 $((NUM_SERVERS - 1))); do
    HOST="${SERVER_HOSTS[$i]}"
    PID="${SERVER_PIDS[$i]}"
    echo "    Server $i on $HOST (PID: $PID)..."
    sshStopCommand "$HOST" "$PID" 2>/dev/null || true
  done
  
  echo "  Waiting for graceful shutdown..."
  sleep 3
  echo "All processes stopped."
  exit 0
}

trap cleanup SIGINT SIGTERM

# --- Wait for client to finish ---
echo "Waiting for client to complete..."

# Poll the client process until it exits
while ssh -o StrictHostKeyChecking=no "$USER@$CLIENT_HOST" "kill -0 $CLIENT_REMOTE_PID" 2>/dev/null; do
  sleep 2
done

echo
echo "Client finished!"
echo "Stopping servers gracefully (sending SIGINT)..."

# Send SIGINT to all servers
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  HOST="${SERVER_HOSTS[$i]}"
  PID="${SERVER_PIDS[$i]}"
  echo "  Sending SIGINT to server $i on $HOST (PID: $PID)..."
  sshStopCommand "$HOST" "$PID" || true
done

echo "Waiting for servers to exit gracefully..."
sleep 3

echo "All processes stopped successfully!"
echo "Done!"
