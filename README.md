# Pre-Req: 

## For no sudo , update bashrc
add this to the end
```
export PATH="$HOME/.local/bin:$PATH"
eval "$(direnv hook bash)"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
export INTEL_OMP_LIB=$(find ~/.local/intel-mkl -name "libiomp5.so" | head -1)
export OMP_PATH=$(dirname "$INTEL_OMP_LIB")
export MKL_LIB_PATH=$(find $HOME/.local/intel-mkl -name "libmkl_core.so" | head -1 | xargs dirname)
export MKL_INCLUDE_PATH=$(find $HOME/.local/intel-mkl -name "mkl.h" | head -1 | xargs dirname)
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

## Install direnv first to make life easier
for non sudo: 
``` 
export $bin_path=$HOME/.local/bin
curl -sfL https://direnv.net/install.sh | bash
```

## Install derecho/cascade
Follow the guide here: https://docs.google.com/document/d/108KxSywDMZ3suJ3kaoqFck7SFpBnq902Huu_4PerFM0/edit?tab=t.0
- create you own [name]_env.sh file and run the files here step by step: https://github.com/aliciayuting/CloudlabSetup/tree/main/installation

`git submodule update --init --recursive --remote`
this downloads all the dependecies to extern, then we have to install some dependencies for these dependencies :(
## install libzmq
follow: https://zeromq.org/download/
```
echo "deb https://download.opensuse.org/repositories/network:/messaging:/zeromq:/git-draft/xUbuntu_22.04/ ./" | sudo tee -a /etc/apt/sources.list
wget https://download.opensuse.org/repositories/network:/messaging:/zeromq:/git-draft/xUbuntu_22.04/Release.key -O- | sudo apt-key add
sudo apt-get install libzmq3-dev

```

```
git clone https://github.com/zeromq/libzmq/
cd libzmq
./autogen.sh
./configure --prefix=/usr/local --enable-drafts
make -j
sudo make install
sudo ldconfig


```


## nholman json


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

### Non sudo 
have to install all the prereq manually, hardest is 
clang-format


```
pip install --user clang-format
which clang-format
```


libaio-dev
```
wget https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz
tar -xzf libaio-libaio-0.3.113.tar.gz
cd libaio-libaio-0.3.113
make prefix=$HOME/.local install
```


gperf
```
# Download gperftools (which provides tcmalloc)
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.15/gperftools-2.15.tar.gz
tar -xzf gperftools-2.15.tar.gz
cd gperftools-2.15

# Configure and build
./configure --prefix=$HOME/.local
make -j$(nproc)
make install
```

boost
https://www.boost.org/doc/user-guide/getting-started.html
```
cd /tmp
git clone https://github.com/boostorg/boost.git -b boost-1.85.0 boost_1_85_0 --depth 1
cd boost_1_85_0
git submodule update --depth 1 --init --recursive


mkdir __build
cd __build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local
cmake --build .
cmake --build . --target install

```



libmkl-full-dev
```
cd ~/workspace
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/47c7d946-fca1-441a-b0df-b094e3f045ea/intel-onemkl-2025.2.0.629_offline.sh

sh ./intel-onemkl-2025.2.0.629_offline.sh -a --silent --cli --eula accept --install-dir $HOME/.local/intel-mkl
# Navigate to the MKL lib directory
cd ~/.local/intel-mkl/mkl/2025.2/lib

# Create the symlink that DiskANN expects
ln -sf libmkl_core.so libmkl_def.so

# Find MKL library path
MKL_LIB_PATH=$(find $HOME/.local/intel-mkl -name "libmkl_core.so" | head -1 | xargs dirname)

# Find MKL include path
MKL_INCLUDE_PATH=$(find $HOME/.local/intel-mkl -name "mkl.h" | head -1 | xargs dirname)

# Find Intel OpenMP path
INTEL_OMP_LIB=$(find ~/.local/intel-mkl -name "libiomp5.so" | head -1)
OMP_PATH=$(dirname "$INTEL_OMP_LIB")

# Verify paths
echo "MKL_LIB_PATH: $MKL_LIB_PATH"
echo "MKL_INCLUDE_PATH: $MKL_INCLUDE_PATH"
echo "OMP_PATH: $OMP_PATH"
```

build diskann
```

# Navigate to DiskANN build directory
mkdir ~/workspace/rdma_anns/extern/DiskANN/build
cd ~/workspace/rdma_anns/extern/DiskANN/build

# Clear any previous CMake cache
rm -rf *

# Configure with all required paths
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_PATH="$MKL_LIB_PATH" \
  -DMKL_INCLUDE_PATH="$MKL_INCLUDE_PATH" \
  -DOMP_PATH="$OMP_PATH" \
  -Wno-dev
  
make -j 
```


## gp-ann
`git submodule update --init --recursive`
sudo
`sudo apt-get install libsparsehash-dev` 

non sudo variety
```
cd /tmp

# Download sparsehash source
wget https://github.com/sparsehash/sparsehash/archive/sparsehash-2.0.4.tar.gz
tar -xzf sparsehash-2.0.4.tar.gz
cd sparsehash-sparsehash-2.0.4

# Configure and build
./configure --prefix=$HOME/.local
make -j$(nproc)
make install
```




tbb
```
cd /tmp
git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j
cmake --install build
```

## parlayann:
`git submodule init`
`git submodule update`



## liburing
```
./configure
make -j
```

# Build

no sudo 
```
cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=OFF -DTEST_UDL1=OFF -DDISK_FS_DISKANN_WRAPPER=OFF -DDISK_FS_DISTRIBUTED=ON -DDISK_KV=OFF -DIN_MEM=OFF -DPQ_KV=OFF -DPQ_FS=ON -DDATA_TYPE=uint8 -DTEST_COMPUTE_PIPELINE=OFF   -DMKL_PATH="$MKL_LIB_PATH" -DMKL_INCLUDE_PATH="$MKL_INCLUDE_PATH" -DOMP_PATH="$OMP_PATH"
```



`cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL2=OFF -DTEST_UDL1=OFF -DDISK_FS_DISKANN_WRAPPER=OFF -DDISK_FS_DISTRIBUTED=ON -DDISK_KV=OFF -DIN_MEM=OFF -DPQ_KV=OFF -DPQ_FS=ON -DDATA_TYPE=uint8 -DTEST_COMPUTE_PIPELINE=OFF`
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


# RDMA

## fractus
compute 20
```
8: roce0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP group default qlen 1000
    link/ether 7c:fe:90:32:76:f7 brd ff:ff:ff:ff:ff:ff
    altname enp69s0f1np1
    inet 192.168.9.20/24 brd 192.168.9.255 scope global roce0
       valid_lft forever preferred_lft forever
    inet6 fe80::7efe:90ff:fe32:76f7/64 scope link
       valid_lft forever preferred_lft forever
```

compute 16
```
10: roce0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP group default qlen 1000
    link/ether 7c:fe:90:32:77:9b brd ff:ff:ff:ff:ff:ff
    altname enp69s0f1np1
    inet 192.168.9.16/24 brd 192.168.9.255 scope global roce0
       valid_lft forever preferred_lft forever
    inet6 fe80::7efe:90ff:fe32:779b/64 scope link
       valid_lft forever preferred_lft forever
```
