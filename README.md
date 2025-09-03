# Pre-Req: 
`git submodule update --init --recursive --remote`
this downloads all the dependecies to extern, then we have to install some dependencies for these dependencies :(
## gcc
need to install gcc-10 and g++-10: 
https://askubuntu.com/questions/1192955/how-to-install-g-10-on-ubuntu-18-04
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10

sudo apt install g++-10
#Remove the previous alternatives
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++

#Define the compiler
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++

#Confirm and update (You can use the default setting)
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```


## parlaylib: 
https://cmuparlay.github.io/parlaylib/installation.html


## catch2: 
Follow: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#installing-catch2-from-git-repository

## Diskann: 
follow https://github.com/microsoft/DiskANN/
- when installing mkl blas, yes to everything
## gp-ann
`sudo apt-get install libsparsehash-dev` 
`git submodule update --init --recursive`
## parlayann:
`git submodule init`
`git submodule update`
# Build



`cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=OFF -DTEST_UDL1=OFF -DDISK_FS_DISKANN_WRAPPER=OFF -DDISK_FS_DISTRIBUTED=ON -DDISK_KV=OFF -DIN_MEM=OFF -DPQ_KV=OFF -DPQ_FS=ON -DDATA_TYPE=uint8`
`cmake --build build -j`

- TEST\_UDL1: run\_benchmark sends queries to udl1 pathname and receives back a GreedySearchQuery with cluster id = 0 and candidate queue is the results of the search. Used to test recall of udl 1
- TEST\_UDL2: run\_benchmark sends queries to udl1 pathname and receives back ANNResult. Used to test udl 2
- IN\_MEM: test the volatile keyvalue store implementation of searching the index. Data is fetched from kvstore instead of being preloaded from a file. Since we don't do in memory search anymore, we should probably delete this sometime in the future. 
- DISK\_FS\_DISKANN\_WRAPPER: used for testing disk search where we load the index from file and just use a thin diskann::PQFlashIndex wrapper to search. This only works for 1 cluster scenario, no communication between clusters. Mainly used for testing and we will probably use this for the shard baseline in the future.
- DISK\_FS\_DISTRIBUTED: our implementation of searching on a hollistic global index. vector embedding + neighbor id is read from file, same way that diskann reads them. PQ data is fetched from volatile kv store to enable cascade get() requests from other clusters for the pq data. If a candidate node during greedy search is not on the server (as determined by a cluster assignment file) then we can send a compute query to the cluster actually containing it to get the distances of its neighbors to the query.
  - PQ_KV
  - PQ_FS
- DISK\_KV: Should be the same idea as the above (currently only works for 1 cluster tho) but the vector embedding and neighbor ids are stored on cascade persistent kvstore instead of on file.
- TEST\_COMPUTE\_PIPELINE: with this enabled, when the distance compute thread receives the compute query, it won't do any computation/read, just return with blank compute result
