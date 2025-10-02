set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/communicator
)
# Try to find ZeroMQ via CMake config first
find_package(ZeroMQ QUIET)

# If not found, try pkg-config
if(NOT ZeroMQ_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZeroMQ REQUIRED libzmq)
    
    # Create an imported target for compatibility
    if(NOT TARGET libzmq)
        add_library(libzmq INTERFACE IMPORTED)
        set_target_properties(libzmq PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ZeroMQ_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${ZeroMQ_LIBRARIES}"
        )
    endif()
endif()


find_package(nlohmann_json REQUIRED)

add_library(communicator communicator.cpp)
target_include_directories(communicator PUBLIC ${COMMON_INCLUDE_DIR})
target_link_libraries(communicator nlohmann_json::nlohmann_json ${ZeroMQ_LIBRARIES})
target_compile_definitions(communicator PUBLIC ZMQ_BUILD_DRAFT_API=1)

add_executable(zmq_p2p_server zmq_p2p_server.cpp)
target_include_directories(zmq_p2p_server PUBLIC ${COMMON_INCLUDE_DIR})
target_link_libraries(zmq_p2p_server communicator ${ZeroMQ_LIBRARIES})
target_compile_definitions(zmq_p2p_server PRIVATE ZMQ_BUILD_DRAFT_API=1)


add_executable(get_version get_version.cpp)
target_include_directories(get_version PUBLIC ${COMMON_INCLUDE_DIR})
target_link_libraries(get_version ${ZeroMQ_LIBRARIES})
