# sift 100k
# ../../benchmark/setup -P /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/disk_index_sift_learn_R32_L50_A1.2 --data_file  /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/sift_learn.fbin -T float        -F Euclidian    -O ~/workspace/rdma_anns/extern/DiskANN/build/data/sift/clusters    -N 3  --query_file /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/sift_query.fbin --gt_file /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/sift_query_learn_gt100 --pq_vectors /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/disk_index_sift_learn_R32_L50_A1.2_pq_compressed.bin --head_index_path /home/nam/workspace/rdma_anns/extern/DiskANN/build/data/sift/disk_index_sift_learn_R32_L50_A1.2_head_index

#bigann 10M
# ../../benchmark/setup -P /home/nam/big-ann-benchmarks/data/bigann/disk_index_bigann_10M_R32_L50_A1.2 --data_file /home/nam/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_10000000 -T uint8        -F Euclidian    -O /home/nam/big-ann-benchmarks/data/bigann/clusters    -N 3  --query_file /home/nam/big-ann-benchmarks/data/bigann/query.public.10K.u8bin --gt_file /home/nam/big-ann-benchmarks/data/bigann/bigann-10M --pq_vectors /home/nam/big-ann-benchmarks/data/bigann/disk_index_bigann_10M_R32_L50_A1.2_pq_compressed.bin --head_index_path /home/nam/big-ann-benchmarks/data/bigann/disk_index_bigann_10M_R32_L50_A1.2_head_index

#bigann 10M
../../benchmark/run_benchmark -Q /home/nam/big-ann-benchmarks/data/bigann/query.public.10K.u8bin -T uint8 -G /home/nam/big-ann-benchmarks/data/bigann/bigann-10M --K 10 --L 20 --num_queries_to_send 10000 --start_node_id 0

