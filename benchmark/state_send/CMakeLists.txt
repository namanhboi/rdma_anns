set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/state_send
)

add_executable(run_benchmark_state_send run_benchmark.cpp)
target_compile_features(run_benchmark_state_send PRIVATE cxx_std_20)
target_include_directories(run_benchmark_state_send PRIVATE ${COMMON_INCLUDE_DIR})
target_link_libraries(run_benchmark_state_send
  ssd_state_send)

target_compile_options(run_benchmark_state_send PRIVATE -g -O0 -Wall)
