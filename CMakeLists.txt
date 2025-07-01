cmake_minimum_required(VERSION 3.31)
project(rdma_anns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(Parlay INTERFACE)
target_include_directories(Parlay INTERFACE
  ParlayANN/parlaylib/include
)

add_library(ParlayANN INTERFACE)
target_include_directories(ParlayANN INTERFACE
  ParlayANN/algorithms
)

target_link_libraries(ParlayANN PUBLIC Parlay)



