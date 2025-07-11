add_executable(setup setup.cpp partition_prebuilt_graph.cpp partition_prebuilt_graph.hpp)
target_compile_features(setup PRIVATE cxx_std_17)
target_include_directories(setup PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_link_libraries(setup PRIVATE ParlayANN derecho::cascade Boost::program_options partition_lib diskann_s aio)



add_custom_command(TARGET setup POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/cfg ${CMAKE_BINARY_DIR}/cfg
  COMMAND ${CMAKE_COMMAND} -E copy    ${CMAKE_CURRENT_SOURCE_DIR}/run_setup.sh        ${CMAKE_CURRENT_BINARY_DIR}/run_setup.sh
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n0/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n1/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n2/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n3/layout.json
    )


