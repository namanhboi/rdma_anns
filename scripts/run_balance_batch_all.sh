mkdir data

echo "This script generates the data for balance_batch and balance_all."

function run_10_10M_batch() {
    {
    NTHREADS=$6
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5 BIGANN index file prefix, $6: num threads
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $6 BIGANN"
    build/benchmark/state_send/run_benchmark_state_send uint8 $5 $NTHREADS $2 ~/big-ann-benchmarks/data/bigann/query.public.10K.u8bin ~/big-ann-benchmarks/data/bigann/bigann-10M $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
    }
}


function run_10_10M_all() {
    {
    NTHREADS=$6
    # $1: top-K (10 in typical), $2: max I/O pipeline width, $3: search mode, $4: mem_L (0 for DiskANN, 10 for Starling and PipeANN)
    # $5 BIGANN index file prefix, $6: num threads
    echo "[REPORT] K $1 BW $2 MODE $3 MEM_L $4 T $6 BIGANN" 
    build_balance_all/benchmark/state_send/run_benchmark_state_send uint8 $5 $NTHREADS $2 ~/big-ann-benchmarks/data/bigann/query.public.10K.u8bin ~/big-ann-benchmarks/data/bigann/bigann-10M $1 l2 $3 $4 10 15 20 25 30 35 40 50 60 80 120 200 400
    }
}


echo "Run balance_batch..."
run_10_10M_batch 10 1 6 0 ~/big-ann-benchmarks/data/bigann/pipeann_10M 15 >> ./data/comparison_balance_batch.txt

echo "Run balance_all..."
run_10_10M_all 10 1 7 0 ~/big-ann-benchmarks/data/bigann/pipeann_10M 15 >> ./data/comparison_balance_all.txt
