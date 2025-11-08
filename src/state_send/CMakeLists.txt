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

add_library(graph_partitioning_utils graph_partitioning_utils.cpp)
target_link_libraries(graph_partitioning_utils PUBLIC partition_lib OpenMP::OpenMP_CXX Boost::program_options)
target_include_directories(graph_partitioning_utils PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(graph_partitioning_utils PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)

find_package(Parlay REQUIRED)


target_compile_features(ssd_state_send PRIVATE cxx_std_20)
target_include_directories(ssd_state_send PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(ssd_state_send PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
  
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
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)

add_library(state_send_client SHARED state_send_client.cpp)
target_link_libraries(state_send_client ssd_state_send Boost::program_options)
# add_executable(state_send_client state_send_client.cpp)
target_compile_options(state_send_client PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)



add_executable(create_scatter_gather_indices create_scatter_gather_indices.cpp)
target_link_libraries(create_scatter_gather_indices ssd_state_send Boost::program_options)
target_compile_options(create_scatter_gather_indices PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(create_state_send_indices create_state_send_indices.cpp)
target_link_libraries(create_state_send_indices ssd_state_send Boost::program_options)
target_compile_options(create_state_send_indices PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ssd_state_send Boost::program_options)
target_compile_options(build_disk_index PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(create_pq_data create_pq_data.cpp)
target_link_libraries(create_pq_data ssd_state_send)
target_compile_options(create_pq_data PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(create_partition_loc_files create_partition_loc_files.cpp)
target_link_libraries(create_partition_loc_files ssd_state_send)
target_compile_options(create_partition_loc_files PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(create_overlap_partition_loc_files create_overlap_partition_loc_files.cpp)
target_link_libraries(create_overlap_partition_loc_files ssd_state_send)
target_compile_options(create_overlap_partition_loc_files PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)



add_executable(convert_old_partition_assignment convert_old_partition_assignment.cpp)
target_link_libraries(convert_old_partition_assignment ssd_state_send)
target_compile_options(convert_old_partition_assignment PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)



add_executable(create_base_file_from_loc_file create_base_file_from_loc_file.cpp)
target_link_libraries(create_base_file_from_loc_file ssd_state_send)
target_compile_options(create_base_file_from_loc_file PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)


add_executable(convert_parlayann_graph_file convert_parlayann_graph_file.cpp)
target_link_libraries(convert_parlayann_graph_file ssd_state_send)
target_compile_options(convert_parlayann_graph_file PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -fopenmp -fopenmp-simd>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fopenmp -fopenmp-simd>
)
