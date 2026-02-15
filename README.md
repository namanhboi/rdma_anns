# How to create the indices

Note that system doesn't support datasets that is extremely high dimensional, data needs to fit into 1 page on disk.
Currently only support l2 and mips

By convention, this repo should be placed in `$HOME/workspace/rdma_anns`

First, we need to build the gp-ann repo. Navigate to `extern/gp-ann`
- by default its l2: `cmake -S. -Bbuild_l2`
- to use mips, need to include `-DMIPS_DISTANCE=ON` when building: `cmake -S. -Bbuild_mips -DMIPS_DISTANCE=ON`
then do `cmake --build build -j`

Then you can partition with the script in `scripts/index_creation/parition.sh`, then you can convert it to the format that is used with `src/state_send/convert_partition_txt_to_bin.cpp`
	

Then, for BatANN (StateSend), we need to build a large index encompassing all points in the dataset using ParlayANN. The script to do this can be found in `scripts/index_creation/create_parlayann_graph.sh`

Then we build the indices for each partition individually with `scripts/index_creation/create_indices.sh`.

Calculate groundtruth with diskann (important for dataset using mips)

# How to run experiments

`scripts/run_experiment.sh` is how you run an experiments. It calls `scripts/setup_exp_vars.sh` which parses the user arguments and sets up the variables in the `run_experiment.sh` script.

The resulting log files will be saved to a folder you specified.

# Build ParlayANN to create the indices
```
cd ~/workspace/rdma_anns
git submodule init
git submodule update
cd extern/ParlayANN/algorithms/vamana
make
```
# Build BatANN: 
## sudo
```
sudo apt update -y
sudo apt-get install -y autoconf automake libtool pkg-config
sudo apt install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libjemalloc-dev


mkdir ~/workspace


cd ~/workspace
sudo apt install libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.31.0/cmake-3.31.0.tar.gz
tar -xvzf cmake-3.31.0.tar.gz
cd cmake-3.31.0
./bootstrap
make -j
sudo make install




cd ~/workspace
git clone https://github.com/zeromq/libzmq/
cd libzmq
./autogen.sh
./configure --prefix=/usr/local --enable-drafts
make -j
sudo make install
sudo ldconfig


cd ~/workspace
git clone https://github.com/cmuparlay/parlaylib
cd parlaylib
mkdir build && cd build
cmake ..
sudo cmake --build . --target install



sudo apt-get install nlohmann-json3-dev


cd ~/workspace
git clone https://github.com/catchorg/Catch2.git
cd Catch2
cmake -B build -S . -DBUILD_TESTING=OFF
sudo cmake --build build/ -j --target install



cd ~/workspace
git clone git@github.com:namanhboi/rdma_anns.git
cd ~/workspace/rdma_anns
git submodule update --init --recursive --remote


cd ~/workspace/rdma_anns/extern/liburing
./configure
make -j



cd ~/workspace/rdma_anns/extern/gp-ann
git submodule update --init --recursive
sudo apt-get install libsparsehash-dev




cd ~/workspace/rdma_anns
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake .. 
sudo cmake --build . --target install



cd ~/workspace/rdma_anns
cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=OFF -DTEST_UDL1=OFF -DDISK_FS_DISKANN_WRAPPER=OFF -DDISK_FS_DISTRIBUTED=ON -DDISK_KV=OFF -DIN_MEM=OFF -DPQ_KV=OFF -DPQ_FS=ON -DDATA_TYPE=uint8 -DTEST_COMPUTE_PIPELINE=OFF -DBALANCE_ALL=OFF -DCMAKE_BUILD_TYPE=RELEASE



cmake --build build/ -j

cd ~/
git clone https://github.com/harsha-simhadri/big-ann-benchmarks
cd ~/big-ann-benchmarks
sudo apt install -y python3-pip
pip install -r requirements_py3.10.txt
python create_dataset.py --dataset bigann-10M

```
## No sudo
setting up .bashrc
```
source ~/.local/intel-mkl/setvars.sh

# Library paths (runtime)
export LD_LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib:$LD_LIBRARY_PATH"

# Build-time library paths
export LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib"

# Include paths
export C_INCLUDE_PATH="$HOME/.local/include"
export CPLUS_INCLUDE_PATH="$HOME/.local/include"

# Package config
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$HOME/.local/lib64/pkgconfig"

# Binaries
export PATH="$HOME/.local/bin:$PATH"

# Aliases
if ! pgrep -u "$USER" emacs > /dev/null; then
    emacs --daemon &
fi
alias ec='emacsclient -nw'

```

```
mkdir ~/workspace
cd ~/workspace
wget https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz
tar -xzf libaio-libaio-0.3.113.tar.gz
cd libaio-libaio-0.3.113
make prefix=$HOME/.local install


cd ~/workspace
git clone https://github.com/zeromq/libzmq/
cd libzmq
./autogen.sh
./configure --prefix=$HOME/.local --enable-drafts
make -j
make install

cd ~/workspace
git clone https://github.com/nlohmann/json
cd json
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local -DJSON_BuildTests=Off
cmake --build . --target install


cd ~/workspace
git clone https://github.com/catchorg/Catch2.git
cd Catch2
cmake -B build -S . -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build/ -j --target install


# Download gperftools (which provides tcmalloc)
cd ~/workspace
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.15/gperftools-2.15.tar.gz
tar -xzf gperftools-2.15.tar.gz
cd gperftools-2.15

# Configure and build
./configure --prefix=$HOME/.local
make -j$(nproc)
make install

cd ~/workspace
git clone https://github.com/boostorg/boost.git -b boost-1.85.0 boost_1_85_0 --depth 1
cd boost_1_85_0
git submodule update --depth 1 --init --recursive


mkdir __build
cd __build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local
cmake --build .
cmake --build . --target install


cd ~/workspace
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/47c7d946-fca1-441a-b0df-b094e3f045ea/intel-onemkl-2025.2.0.629_offline.sh

sh ./intel-onemkl-2025.2.0.629_offline.sh -a --silent --cli --eula accept --install-dir $HOME/.local/intel-mkl
# Navigate to the MKL lib directory
cd ~/.local/intel-mkl/mkl/2025.2/lib
# Create the symlink that DiskANN expects
ln -sf libmkl_core.so libmkl_def.so


cd $HOME/.local/intel-mkl/mkl/2025.2/include
ln -s mkl_cblas.h cblas.h



pip install --user clang-format
which clang-format




cd ~/workspace
git clone https://github.com/cmuparlay/parlaylib
cd parlaylib
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local
cmake --build . --target install



cd ~/workspace
git clone https://github.com/catchorg/Catch2.git
cd Catch2
cmake -B build -S . -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local
cmake --build build/ -j --target install





cd ~/workspace
git clone git@github.com:namanhboi/rdma_anns.git
cd ~/workspace/rdma_anns
git submodule update --init --recursive --remote

cd ~/workspace/rdma_anns/extern/liburing
./configure
make -j


cd ~/workspace/rdma_anns
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local
cmake --build . -j --target install


cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=OFF -DTEST_UDL1=OFF -DDISK_FS_DISKANN_WRAPPER=OFF -DDISK_FS_DISTRIBUTED=ON -DDISK_KV=OFF -DIN_MEM=OFF -DPQ_KV=OFF -DPQ_FS=ON -DDATA_TYPE=uint8 -DTEST_COMPUTE_PIPELINE=OFF -DBALANCE_ALL=OFF -DCMAKE_BUILD_TYPE=RELEASE -DMKL_ROOT=$MKLROOT   -DCMAKE_CXX_FLAGS="-I$HOME/.local/intel-mkl/mkl/2025.2/include"

cmake --build build -j

```

- TEST\_UDL2: run\_benchmark sends queries to udl1 pathname and receives back ANNResult. Used to test udl 2
- IN\_MEM: test the volatile keyvalue store implementation of searching the index. Data is fetched from kvstore instead of being preloaded from a file. Since we don't do in memory search anymore, we should probably delete this sometime in the future. 
- DISK\_FS\_DISKANN\_WRAPPER: used for testing disk search where we load the index from file and just use a thin diskann::PQFlashIndex wrapper to search. This only works for 1 cluster scenario, no communication between clusters. Mainly used for testing and we will probably use this for the shard baseline in the future.
- DISK\_FS\_DISTRIBUTED: our implementation of searching on a hollistic global index. vector embedding + neighbor id is read from file, same way that diskann reads them. PQ data is fetched from volatile kv store to enable cascade get() requests from other clusters for the pq data. If a candidate node during greedy search is not on the server (as determined by a cluster assignment file) then we can send a compute query to the cluster actually containing it to get the distances of its neighbors to the query.
  - PQ_KV
  - PQ_FS
- DISK\_KV: Should be the same idea as the above (currently only works for 1 cluster tho) but the vector embedding and neighbor ids are stored on cascade persistent kvstore instead of on file.
- TEST\_COMPUTE\_PIPELINE: with this enabled, when the distance compute thread receives the compute query, it won't do any computation/read, just return with blank compute result

# DATASET info
- bigann: l2, 128, uint8
- msspacev: l2, 100, int8
- deep: l2, 96, float
- text2image: mips, 200, float
- openaiarxiv: cosine, 1536, float
