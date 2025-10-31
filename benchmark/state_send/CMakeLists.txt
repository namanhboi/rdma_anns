set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/state_send
  
)

# add_executable(run_benchmark_state_send_local run_benchmark_state_send_local.cpp)
# target_compile_features(run_benchmark_state_send_local PRIVATE cxx_std_20)
# target_include_directories(run_benchmark_state_send_local PRIVATE ${COMMON_INCLUDE_DIR})
# target_link_libraries(run_benchmark_state_send_local
#   ssd_state_send
# Boost::program_options)

# target_compile_options(run_benchmark_state_send_local PRIVATE -g -O0 -Wall)


find_package(nlohmann_json REQUIRED)
add_executable(run_benchmark_state_send_tcp run_benchmark_state_send_tcp.cpp)
target_compile_features(run_benchmark_state_send_tcp PRIVATE cxx_std_20)
target_include_directories(run_benchmark_state_send_tcp PRIVATE ${COMMON_INCLUDE_DIR})
target_link_libraries(run_benchmark_state_send_tcp
  state_send_client
  nlohmann_json::nlohmann_json
  Boost::program_options
)

target_compile_options(run_benchmark_state_send_tcp PRIVATE -g -O0 -Wall)
