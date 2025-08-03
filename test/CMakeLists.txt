find_package(Catch2 3 REQUIRED)

add_executable(serialize_test serialize_test.cpp)
target_include_directories(serialize_test PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(serialize_test PRIVATE Catch2::Catch2WithMain diskann aio derecho derecho::cascade pthread)
target_compile_options(serialize_test PRIVATE -g -O0 -Wall)
