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

add_subdirectory(extern/gp-ann)
add_subdirectory(extern/DiskANN)
add_subdirectory(benchmark)

