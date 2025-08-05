add_executable(setup setup.cpp data_loading_utils.cpp data_loading_utils.hpp)
target_compile_features(setup PRIVATE cxx_std_20)
target_include_directories(setup PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(setup PRIVATE ParlayANN derecho::cascade Boost::program_options partition_lib diskann aio)
# target_compile_options(setup PRIVATE -g -O0 -Wall)



add_executable(run_benchmark run_benchmark.cpp benchmark_dataset.hpp)
target_compile_features(run_benchmark PRIVATE cxx_std_20)
target_include_directories(run_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(run_benchmark PRIVATE ParlayANN derecho::cascade Boost::program_options diskann aio)
target_compile_options(run_benchmark PRIVATE -O3)
