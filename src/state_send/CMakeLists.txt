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
)

add_library(ssd_state_send SHARED
  ${CPP_FILES}
)
find_package(OpenMP REQUIRED)

target_compile_features(ssd_state_send PRIVATE cxx_std_20)
target_include_directories(ssd_state_send PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(ssd_state_send PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops>
)
target_link_directories(ssd_state_send PUBLIC
	${PROJECT_SOURCE_DIR}/extern/liburing/src
)
target_link_libraries(ssd_state_send uring communicator OpenMP::OpenMP_CXX)

if(BALANCE_ALL)
    target_compile_definitions(ssd_state_send PRIVATE BALANCE_ALL)
endif()

add_executable(state_send_server state_send_server.cpp)
target_link_libraries(state_send_server ssd_state_send)
target_compile_options(state_send_server PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops>
)

add_library(state_send_client SHARED state_send_client.cpp)
target_link_libraries(state_send_client ssd_state_send)
# add_executable(state_send_client state_send_client.cpp)
target_compile_options(state_send_client PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall>
  $<$<CONFIG:Release>:-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops>
)

