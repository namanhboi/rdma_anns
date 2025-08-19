set(UDL_COMMON_LIBS derecho derecho::cascade pthread)


add_library(head_index_search_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/head_index_search_udl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/head_index_search_udl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/udl_path_and_index.hpp
)
target_include_directories(head_index_search_udl PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_compile_definitions(head_index_search_udl PRIVATE DATA_TYPE="float")

target_link_libraries(head_index_search_udl ${UDL_COMMON_LIBS} diskann aio)
# target_compile_options(head_index_search_udl PRIVATE "-O3")


add_library(global_search_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/global_search_udl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/global_search_udl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/get_request_manager.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/in_mem_search_index.hpp
)
target_include_directories(global_search_udl PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_compile_definitions(global_search_udl PRIVATE DATA_TYPE="float" GLOBAL_BASELINE)
target_link_libraries(global_search_udl ${UDL_COMMON_LIBS} diskann aio)
# target_compile_options(global_search_udl PRIVATE "-O3")



add_custom_target(update_config ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/cfg ${CMAKE_BINARY_DIR}/cfg
  
  # Create node directories if they don't exist
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cfg/n0
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cfg/n1
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cfg/n2
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cfg/n3
  
  # Create cross-platform symbolic links for layout.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/layout.json
    ${CMAKE_BINARY_DIR}/cfg/n0/layout.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/layout.json
    ${CMAKE_BINARY_DIR}/cfg/n1/layout.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/layout.json
    ${CMAKE_BINARY_DIR}/cfg/n2/layout.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/layout.json
    ${CMAKE_BINARY_DIR}/cfg/n3/layout.json
    
  # Create symbolic links for dfgs.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/dfgs.json
    ${CMAKE_BINARY_DIR}/cfg/n0/dfgs.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/dfgs.json
    ${CMAKE_BINARY_DIR}/cfg/n1/dfgs.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/dfgs.json
    ${CMAKE_BINARY_DIR}/cfg/n2/dfgs.json
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/dfgs.json
    ${CMAKE_BINARY_DIR}/cfg/n3/dfgs.json
    
  # Create symbolic links for udl_dlls.cfg
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
    ${CMAKE_BINARY_DIR}/cfg/n0/udl_dlls.cfg
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
    ${CMAKE_BINARY_DIR}/cfg/n1/udl_dlls.cfg
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
    ${CMAKE_BINARY_DIR}/cfg/n2/udl_dlls.cfg
  COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
    ${CMAKE_BINARY_DIR}/cfg/n3/udl_dlls.cfg
    
  COMMENT "Setting up configuration files and symbolic links"
  VERBATIM
)
