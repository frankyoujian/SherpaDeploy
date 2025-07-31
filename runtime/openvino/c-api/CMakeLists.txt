include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-openvino-c-api c-api.cc)
target_link_libraries(sherpa-openvino-c-api sherpa-openvino-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-openvino-c-api PRIVATE SHERPA_DEPLOY_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-openvino-c-api PRIVATE SHERPA_DEPLOY_BUILD_MAIN_LIB=1)
endif()

install(TARGETS sherpa-openvino-c-api DESTINATION lib)

install(FILES c-api.h
  DESTINATION include/sherpa-openvino/c-api
)
