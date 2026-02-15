pkill -2 -f 'state_send_server' 2>/dev/null
pkill -2 -f 'run_benchmark_state_send_tcp' 2>/dev/null
sleep 1
# If still running, force kill 
pkill -9 -f 'state_send_server' 2>/dev/null
pkill -9 -f 'run_benchmark_state_send_tcp' 2>/dev/null
