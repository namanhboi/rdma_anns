set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/state_send
  ${PROJECT_SOURCE_DIR}/extern/liburing/src/include
  ${PROJECT_SOURCE_DIR}/include/communicator
)

set (CPP_FILES
  distance.cpp
  linux_aligned_file_reader.cpp
  search_state.cpp
  ssd_partition_index.cpp
  utils.cpp
  aux_utils.cpp
  search_thread.cpp
  types.cpp
  query_buf.cpp
  index.cpp
  partition_and_pq.cpp
  math_utils.cpp
  disk_utils.cpp
  distributedann_impl.cpp
)


add_library(ssd_state_send SHARED
  ${CPP_FILES}
)
find_package(spdlog REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "spdlog_FOUND: ${spdlog_FOUND}")
message(STATUS "spdlog_DIR: ${spdlog_DIR}")
message(STATUS "spdlog_INCLUDE_DIRS: ${spdlog_INCLUDE_DIRS}")
message(STATUS "spdlog_LIBRARIES: ${spdlog_LIBRARIES}")

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()
	
find_package(BLAS)
if(BLAS_FOUND)
  message(STATUS "BLAS found")
  message(${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS is required but was not found.")
endif()


set(DEBUG_CXX_FLAGS -g -O0 -Wall -fopenmp -fopenmp-simd -march=native -mavx2 -mfma -msse2)
set(RELEASE_CXX_FLAGS -g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd -mavx2 -mfma -msse2)

add_library(graph_partitioning_utils graph_partitioning_utils.cpp)
target_link_libraries(graph_partitioning_utils PUBLIC partition_lib OpenMP::OpenMP_CXX Boost::program_options)
target_include_directories(graph_partitioning_utils PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(graph_partitioning_utils PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)

find_package(Parlay REQUIRED)


target_compile_features(ssd_state_send PRIVATE cxx_std_20)
target_include_directories(ssd_state_send PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(ssd_state_send PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
  
)
target_link_directories(ssd_state_send PUBLIC
	${PROJECT_SOURCE_DIR}/extern/liburing/src
)
target_link_libraries(ssd_state_send uring communicator OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} graph_partitioning_utils Boost::program_options spdlog::spdlog_header_only Parlay::parlay)

if(BALANCE_ALL)
    target_compile_definitions(ssd_state_send PRIVATE BALANCE_ALL)
endif()



add_executable(state_send_server state_send_server.cpp)
target_link_libraries(state_send_server ssd_state_send Boost::program_options)
target_compile_options(state_send_server PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)

add_library(state_send_client SHARED state_send_client.cpp)
target_link_libraries(state_send_client ssd_state_send Boost::program_options)
# add_executable(state_send_client state_send_client.cpp)
target_compile_options(state_send_client PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)




add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ssd_state_send Boost::program_options)
target_compile_options(build_disk_index PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)


add_executable(create_pq_data create_pq_data.cpp)
target_link_libraries(create_pq_data ssd_state_send)
target_compile_options(create_pq_data PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)


add_executable(create_partition_loc_files create_partition_loc_files.cpp)
target_link_libraries(create_partition_loc_files ssd_state_send)
target_compile_options(create_partition_loc_files PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)


add_executable(create_overlap_partition_loc_files create_overlap_partition_loc_files.cpp)
target_link_libraries(create_overlap_partition_loc_files ssd_state_send)
target_compile_options(create_overlap_partition_loc_files PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)



add_executable(convert_old_partition_assignment convert_old_partition_assignment.cpp)
target_link_libraries(convert_old_partition_assignment ssd_state_send)
target_compile_options(convert_old_partition_assignment PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)



add_executable(create_base_file_from_loc_file create_base_file_from_loc_file.cpp)
target_link_libraries(create_base_file_from_loc_file ssd_state_send)
target_compile_options(create_base_file_from_loc_file PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)


add_executable(convert_parlayann_graph_file convert_parlayann_graph_file.cpp)
target_link_libraries(convert_parlayann_graph_file ssd_state_send)
target_compile_options(convert_parlayann_graph_file PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)

add_executable(gen_random_slice gen_random_slice.cpp)
target_link_libraries(gen_random_slice ssd_state_send)
target_compile_options(gen_random_slice PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)



add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ssd_state_send)
target_compile_options(build_memory_index PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)



add_executable(build_disk_index_from_bin_graph build_disk_index_from_bin_graph.cpp)
target_link_libraries(build_disk_index_from_bin_graph ssd_state_send)
target_compile_options(build_disk_index_from_bin_graph PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)

add_executable(create_graph_file_from_disk_index create_graph_file_from_disk_index.cpp)
target_link_libraries(create_graph_file_from_disk_index ssd_state_send)
target_compile_options(create_graph_file_from_disk_index PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)


add_executable(create_partition_graph_file create_partition_graph_file.cpp)
target_link_libraries(create_partition_graph_file ssd_state_send)
target_compile_options(create_partition_graph_file PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>
)
