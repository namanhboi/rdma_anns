set(UDL_COMMON_LIBS derecho derecho::cascade pthread)


add_library(head_index_search_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/head_index_search_udl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/head_index_search_udl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.cpp
)
target_include_directories(head_index_search_udl PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_compile_definitions(head_index_search_udl PRIVATE DATA_TYPE="float")

target_link_libraries(head_index_search_udl ${UDL_COMMON_LIBS} diskann aio)
target_compile_options(head_index_search_udl PRIVATE "-O3")


add_library(global_search_udl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/global_search_udl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/global_search_udl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/serialize_utils.cpp)
target_include_directories(global_search_udl PRIVATE ${CMAKE_SOURCE_DIR}/extern/DiskANN/include) # needed to include this to access the methods of diskann library
target_compile_definitions(global_search_udl PRIVATE DATA_TYPE="float")

target_link_libraries(global_search_udl ${UDL_COMMON_LIBS} diskann aio)
target_compile_options(global_search_udl PRIVATE "-O3")


add_custom_command(TARGET head_index_search_udl POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/cfg ${CMAKE_BINARY_DIR}/cfg
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n0/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n1/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n2/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/layout.json
  ${CMAKE_BINARY_DIR}/cfg/n3/layout.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/dfgs.json
  ${CMAKE_BINARY_DIR}/cfg/n0/dfgs.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/dfgs.json
  ${CMAKE_BINARY_DIR}/cfg/n1/dfgs.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/dfgs.json
  ${CMAKE_BINARY_DIR}/cfg/n2/dfgs.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/dfgs.json
  ${CMAKE_BINARY_DIR}/cfg/n3/dfgs.json
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
  ${CMAKE_BINARY_DIR}/cfg/n0/udl_dlls.cfg
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
  ${CMAKE_BINARY_DIR}/cfg/n1/udl_dlls.cfg
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
  ${CMAKE_BINARY_DIR}/cfg/n2/udl_dlls.cfg
  COMMAND ln -sf ${CMAKE_BINARY_DIR}/cfg/udl_dlls.cfg
  ${CMAKE_BINARY_DIR}/cfg/n3/udl_dlls.cfg    
)
