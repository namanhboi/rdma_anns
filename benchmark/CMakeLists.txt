add_executable(setup setup.cpp data_loading_utils.cpp data_loading_utils.hpp)
target_compile_features(setup PRIVATE cxx_std_20)
target_include_directories(setup PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(setup PRIVATE ParlayANN derecho::cascade Boost::program_options partition_lib diskann aio)
# target_compile_definitions(setup PRIVATE TEST_UDL1)
target_compile_options(setup PRIVATE -g -O0 -Wall)


add_executable(create_clusters create_clusters.cpp data_loading_utils.cpp data_loading_utils.hpp)
target_compile_features(create_clusters PRIVATE cxx_std_20)
target_include_directories(create_clusters PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(create_clusters PRIVATE ParlayANN derecho::cascade Boost::program_options partition_lib diskann aio)
# target_compile_definitions(setup PRIVATE TEST_UDL1)
target_compile_options(create_clusters PRIVATE -g -O0 -Wall)

add_executable(run_benchmark run_benchmark.cpp benchmark_dataset.hpp)
target_compile_features(run_benchmark PRIVATE cxx_std_20)
target_include_directories(run_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(run_benchmark PRIVATE ParlayANN derecho::cascade Boost::program_options diskann aio)
# target_compile_definitions(run_benchmark PRIVATE TEST_UDL1)
target_compile_options(run_benchmark PRIVATE -g -O0 -Wall)

add_executable(prep_data prep_data.cpp data_loading_utils.cpp data_loading_utils.hpp)
target_compile_features(prep_data PRIVATE cxx_std_20)
target_include_directories(prep_data PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(prep_data PRIVATE ParlayANN derecho::cascade Boost::program_options partition_lib diskann aio)
# target_compile_definitions(setup PRIVATE TEST_UDL1)
target_compile_options(prep_data PRIVATE -g -O0 -Wall)
