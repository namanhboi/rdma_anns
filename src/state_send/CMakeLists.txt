set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/state_send
  ${PROJECT_SOURCE_DIR}/extern/liburing/src/include
)

set (CPP_FILES
  distance.cpp
  io_submission_thread.cpp
  linux_aligned_file_reader.cpp
  search_state.cpp
  ssd_partition_index.cpp
  utils.cpp
  search_thread.cpp
)
add_library(ssd_state_send SHARED
  ${CPP_FILES}
)

target_compile_features(ssd_state_send PRIVATE cxx_std_20)
target_include_directories(ssd_state_send PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(ssd_state_send PRIVATE -g -O0 -Wall)
target_link_directories(ssd_state_send PUBLIC
	${PROJECT_SOURCE_DIR}/extern/liburing/src
)
target_link_libraries(ssd_state_send uring)
