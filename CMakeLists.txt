cmake_minimum_required(VERSION 3.31)
project(rdma_anns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(Parlay INTERFACE)
target_include_directories(Parlay INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/ParlayANN/parlaylib/include
)

add_library(ParlayANN INTERFACE)
target_include_directories(ParlayANN INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/ParlayANN/algorithms
)

target_link_libraries(ParlayANN INTERFACE Parlay)


find_package(cascade CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
message(STATUS "Boost_FOUND: ${Boost_FOUND}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

option(TEST_UDL1 "Enable TEST_UDL1 functionality" OFF)
option(TEST_UDL2 "Enable TEST_UDL2 functionality" OFF)

option(IN_MEM "Search udl 2 with in memory kv store based global search" OFF)
option(DISK_FS_DISKANN_WRAPPER "Search udl 2 with ssd based search where data is loaded from file system" OFF)
option(DISK_FS_DISTRIBUTED "Search udl 2 with ssd based search where data is loaded from file system" OFF)
option(DISK_KV "Search udl 2 with ssd based search where data is loaded from kv store" OFF)

if(TEST_UDL1)
    add_compile_definitions(TEST_UDL1)
endif()

if(TEST_UDL2)
    add_compile_definitions(TEST_UDL2)
endif()

if(IN_MEM)
    add_compile_definitions(IN_MEM)
endif()

if(DISK_FS_DISKANN_WRAPPER)
    add_compile_definitions(DISK_FS_DISKANN_WRAPPER)
endif()

if(DISK_FS_DISTRIBUTED)
    add_compile_definitions(DISK_FS_DISTRIBUTED)
endif()

if(PQ_KV)
    add_compile_definitions(PQ_KV)
endif()
if(PQ_FS)
    add_compile_definitions(PQ_FS)
endif()

if(DISK_KV)
    add_compile_definitions(DISK_KV)
endif()


if(TEST_COMPUTE_PIPELINE)
    add_compile_definitions(TEST_COMPUTE_PIPELINE)
endif()


# Allow DATA_TYPE to be set from command line
if(NOT DEFINED DATA_TYPE)
    set(DATA_TYPE "float")  # default
endif()

add_subdirectory(extern/gp-ann)
add_subdirectory(extern/DiskANN)
add_subdirectory(benchmark)
add_subdirectory(src)
add_subdirectory(test)

