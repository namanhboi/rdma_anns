cmake_minimum_required(VERSION 3.29)
project(rdma_anns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

  # Then, for the runtime.
  set(CPU_SUPPORTS_AVX512 FALSE)
  set(CPU_SUPPORTS_AVX2 FALSE)

  try_run(
    RUN_RESULT
    COMPILE_RESULT
    "${CMAKE_BINARY_DIR}"
    SOURCES "${PROJECT_SOURCE_DIR}/test/test_cpu.cpp"
    COMPILE_OUTPUT_VARIABLE COMPILE_OUT
    RUN_OUTPUT_VARIABLE RUN_OUT
  )


  if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
    set(CPU_SUPPORTS_AVX512 TRUE)
    message(STATUS "CPU supports AVX512")
  elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
    set(CPU_SUPPORTS_AVX2 TRUE)
    message(STATUS "CPU supports AVX2")
  else()
    message(STATUS "CPU does not support AVX512 or detection failed")
  endif()

  if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU supports -mavx512f. Enabling AVX512F.")
    add_definitions(-DUSE_AVX512)
  elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
    message(STATUS "Compiler and CPU supports -mavx2. Enabling AVX2.")
    add_definitions(-DUSE_AVX2)
  else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
  endif()
endif()


include_directories(
  ${PROJECT_SOURCE_DIR}/include/concurrentqueue
)

add_library(Parlay INTERFACE)
target_include_directories(Parlay INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/ParlayANN/parlaylib/include
)

add_library(ParlayANN INTERFACE)
target_include_directories(ParlayANN INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/ParlayANN/algorithms
)

target_link_libraries(ParlayANN INTERFACE Parlay)


# find_package(cascade CONFIG REQUIRED)
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
option(USE_RDMA "whether to use rdma or not" OFF)

set(USER_DEFINITIONS
  TEST_UDL1
  TEST_UDL2
  IN_MEM
  DISK_FS_DISKANN_WRAPPER
  DISK_FS_DISTRIBUTED
  DISK_KV
  PQ_KV
  PQ_FS
  USE_RDMA
)

foreach(_def ${USER_DEFINITIONS})
    if(${_def})
        add_compile_definitions(${_def})
    endif()
endforeach()


if(USE_RDMA)
  add_compile_definitions(USE_RDMA)
endif()

# Allow DATA_TYPE to be set from command line
if(NOT DEFINED DATA_TYPE)
  set(DATA_TYPE "float")  # default
endif()

add_subdirectory(extern/gp-ann)
# add_subdirectory(extern/ParlayANN)
# add_subdirectory(extern/DiskANN)
# add_subdirectory(src/task_push)
add_subdirectory(src/communicator)
add_subdirectory(src/state_send)
# add_subdirectory(benchmark/task_push)
add_subdirectory(benchmark/state_send)
add_subdirectory(benchmark/communicator)
# add_subdirectory(test)
# add_subdirectory(dummy_udl/src)
