set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/communicator
)
add_executable(zmq_ack_rtt zmq_ack_rtt.cpp)
target_include_directories(zmq_ack_rtt PUBLIC ${COMMON_INCLUDE_DIR})
target_link_libraries(zmq_ack_rtt communicator)

