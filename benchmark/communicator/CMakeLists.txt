set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/communicator
)
add_executable(zmq_rtt_local zmq_rtt_local.cpp)
target_include_directories(zmq_rtt_local PUBLIC ${COMMON_INCLUDE_DIR})
target_link_libraries(zmq_rtt_local communicator)

add_executable(zmq_ack zmq_ack.cpp)
target_link_libraries(zmq_ack communicator)

add_executable(zmq_sender zmq_sender.cpp)
target_link_libraries(zmq_sender communicator)
