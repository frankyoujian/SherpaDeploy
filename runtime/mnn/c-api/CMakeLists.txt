include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-mnn-c-api c-api.cc)
target_link_libraries(sherpa-mnn-c-api sherpa-deploy-mnn-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-mnn-c-api PRIVATE SHERPA_DEPOLY_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-mnn-c-api PRIVATE SHERPA_DEPOLY_BUILD_MAIN_LIB=1)
endif()

install(TARGETS sherpa-mnn-c-api DESTINATION lib)

install(FILES c-api.h
  DESTINATION include/sherpa-mnn/c-api
)
