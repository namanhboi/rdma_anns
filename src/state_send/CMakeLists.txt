set(COMMON_INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include/state_send
  ${PROJECT_SOURCE_DIR}/extern/liburing/src/include
)


file(GLOB CPP_FILES *.cpp)
add_library(ssd_state_send SHARED
  ${CPP_FILES}
)

target_compile_features(ssd_state_send PRIVATE cxx_std_20)
target_include_directories(ssd_state_send PUBLIC ${COMMON_INCLUDE_DIR})
target_compile_options(ssd_state_send PRIVATE -g -O0 -Wall)
target_link_directories(ssd_state_send PUBLIC
	${PROJECT_SOURCE_DIR}/extern/liburing/src
)
target_link_libraries(ssd_state_send uring)
