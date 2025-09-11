set(UDL_COMMON_LIBS derecho derecho::cascade pthread)

add_library(dummy_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/dummy_udl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dummy_udl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/udl_path_and_index.hpp
)


set(LOG_DUMMY_BATCH_SEND_START 10000)
set(LOG_DUMMY_BATCH_SEND_END 10001)
set(LOG_DUMMY_HANDLER_START 10010)
set(LOG_DUMMY_HANDLER_END 10011)

target_compile_definitions(dummy_udl PRIVATE
  LOG_DUMMY_BATCH_SEND_START=${LOG_DUMMY_BATCH_SEND_START}
  LOG_DUMMY_BATCH_SEND_END=${LOG_DUMMY_BATCH_SEND_END}
  LOG_DUMMY_HANDLER_START=${LOG_DUMMY_HANDLER_START}
  LOG_DUMMY_HANDLER_END=${LOG_DUMMY_HANDLER_END}
)


target_link_libraries(dummy_udl ${UDL_COMMON_LIBS})
# Function to create configuration setup for a given config type
function(setup_config_target target_name source_dir dest_dir)
  add_custom_target(${target_name} ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../${source_dir} ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}
    
    # Create node directories if they don't exist
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/n0
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/n1
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/n2
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/n3
    
    COMMENT "Setting up configuration files and symbolic links for ${dest_dir}"
    VERBATIM
  )
  
  # List of files to create symbolic links for
  set(CONFIG_FILES layout.json dfgs.json udl_dlls.cfg)
  
  # Create symbolic links for each config file and each node
  foreach(config_file ${CONFIG_FILES})
    foreach(node_id RANGE 0 3)
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink 
          ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/${config_file}
          ${CMAKE_CURRENT_BINARY_DIR}/../${dest_dir}/n${node_id}/${config_file}
      )
    endforeach()
  endforeach()
endfunction()

# Create both configuration targets
setup_config_target(update_config_dummy cfg cfg)
setup_config_target(update_config_rdma_dummy cfg_rdma cfg_rdma)



add_library(dummy_no_deserialize_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/dummy_no_deserialize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dummy_no_deserialize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/udl_path_and_index.hpp
)


target_compile_definitions(dummy_no_deserialize_udl PRIVATE
  LOG_DUMMY_BATCH_SEND_START=${LOG_DUMMY_BATCH_SEND_START}
  LOG_DUMMY_BATCH_SEND_END=${LOG_DUMMY_BATCH_SEND_END}
  LOG_DUMMY_HANDLER_START=${LOG_DUMMY_HANDLER_START}
  LOG_DUMMY_HANDLER_END=${LOG_DUMMY_HANDLER_END}
)

add_executable(run_dummy_benchmark run_benchmark.cpp serialize_utils.hpp benchmark_client.hpp)
target_compile_features(run_dummy_benchmark PRIVATE cxx_std_20)
target_link_libraries(run_dummy_benchmark PRIVATE ${UDL_COMMON_LIBS} Boost::program_options)
target_compile_options(run_dummy_benchmark PRIVATE -g -O0 -Wall)

add_executable(run_dummy_no_deserialize_benchmark run_benchmark_no_deserialize.cpp serialize_utils.hpp)
target_compile_features(run_dummy_no_deserialize_benchmark PRIVATE cxx_std_20)
target_link_libraries(run_dummy_no_deserialize_benchmark PRIVATE ${UDL_COMMON_LIBS} Boost::program_options)
target_compile_options(run_dummy_no_deserialize_benchmark PRIVATE -g -O0 -Wall)

setup_config_target(update_config_dummy_no_deserialize cfg_no_deserialize cfg_no_deserialize)
setup_config_target(update_config_rdma_dummy_no_deserialize cfg_no_deserialize_rdma cfg_no_deserialize_rdma)

