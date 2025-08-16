To build, 
`cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=ON -DTEST_UDL1=OFF -DDISK_FS=ON -DISK_KV=OFF -DIN_MEM=OFF`
`cmake --build build -j`

- TEST\_UDL1: run\_benchmark sends queries to udl1 pathname and receives back a GreedySearchQuery with cluster id = 0 and candidate queue is the results of the search. Used to test recall of udl 1
- TEST\_UDL2: run\_benchmark sends queries to udl1 pathname and receives back ANNResult. Used to test udl 2
- IN\_MEM: test the volatile keyvalue store implementation of searching the index. Data is fetched from kvstore instead of being preloaded from a file. Since we don't do in memory search anymore, we should probably delete this sometime in the future. 
- DISK\_FS\_DISKANN\_WRAPPER: used for testing disk search where we load the index from file and just use a thin diskann::PQFlashIndex wrapper to search. This only works for 1 cluster scenario, no communication between clusters. Mainly used for testing and we will probably use this for the shard baseline in the future.
- DISK\_FS\_DISTRIBUTED: our implementation of searching on a hollistic global index. vector embedding + neighbor id is read from file, same way that diskann reads them. PQ data is fetched from volatile kv store to enable cascade get() requests from other clusters for the pq data. If a candidate node during greedy search is not on the server (as determined by a cluster assignment file) then we can send a compute query to the cluster actually containing it to get the distances of its neighbors to the query.
- DISK\_KV: Should be the same idea as the above (currently only works for 1 cluster tho) but the vector embedding and neighbor ids are stored on cascade persistent kvstore instead of on file.
