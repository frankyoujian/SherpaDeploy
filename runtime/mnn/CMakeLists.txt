include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_mnn_core_srcs
  ${CMAKE_SOURCE_DIR}/runtime/core/context-graph.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/endpoint.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/file-utils.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/hypothesis.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/resample.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/symbol-table.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/wave-reader.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/wave-writer.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/features.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/text-utils.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/utils.cc
  model.cc
  zipformer-model.cc
  mnn-utils.cc
  stream.cc
  recognizer.cc
  decoder.cc
  greedy-search-decoder.cc
  modified-beam-search-decoder.cc
)

#list(APPEND sherpa_mnn_core_srcs
#  ${CMAKE_SOURCE_DIR}/runtime/core/circular-buffer.cc
#  silero-vad-model-config.cc
#  silero-vad-model.cc
#  voice-activity-detector.cc
#)

# Always static build so that there is a single shared library: SherpaDeploy-mnn-c-api.dll
add_library(sherpa-deploy-mnn-core STATIC ${sherpa_mnn_core_srcs})

# MNN project uses include_directories in its own code, it does not use target_include_directories like NCNN
# so external project has no access to its headers path, need to include its headers path manually
get_property(MNN_HEADERS_DIR GLOBAL PROPERTY MNN_HEADERS_DIR)
message(STATUS "MNN_HEADERS_DIR is ${MNN_HEADERS_DIR}")
include_directories(${MNN_HEADERS_DIR})

target_link_libraries(sherpa-deploy-mnn-core PUBLIC kaldi-native-fbank-core MNN ssentencepiece_core)

if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-deploy-mnn-core DESTINATION ..)
else()
  install(TARGETS sherpa-deploy-mnn-core DESTINATION lib)
endif()

add_subdirectory(c-api)

if(NOT SHERPA_ONNX_ENABLE_PYTHON)
  if(SHERPA_DEPLOY_ENABLE_MNN)
    add_executable(sherpa-mnn sherpa-mnn.cc)
    target_link_libraries(sherpa-mnn PRIVATE sherpa-deploy-mnn-core)
    install(TARGETS sherpa-mnn DESTINATION bin)
#
#    add_executable(sherpa-ncnn-vad sherpa-ncnn-vad.cc)
#    target_link_libraries(sherpa-ncnn-vad PRIVATE sherpa-ncnn-core)
#    install(TARGETS sherpa-ncnn-vad DESTINATION bin)
#
#    if(SHERPA_NCNN_HAS_ALSA)
#      add_executable(sherpa-ncnn-alsa sherpa-ncnn-alsa.cc alsa.cc)
#      target_link_libraries(sherpa-ncnn-alsa PRIVATE sherpa-ncnn-core)
#
#      if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
#        target_link_libraries(sherpa-ncnn-alsa PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
#      else()
#        target_link_libraries(sherpa-ncnn-alsa PRIVATE asound)
#      endif()
#      install(TARGETS sherpa-ncnn-alsa DESTINATION bin)
#    endif()
#
    if(SHERPA_ONNX_ENABLE_PORTAUDIO)
      add_executable(sherpa-mnn-microphone
        sherpa-mnn-microphone.cc
        ${CMAKE_SOURCE_DIR}/runtime/core/microphone.cc
      )

      set(PA_LIB portaudio_static)

      target_link_libraries(sherpa-mnn-microphone PRIVATE ${PA_LIB} sherpa-mnn-c-api)

      install(TARGETS sherpa-mnn-microphone DESTINATION bin)
    endif()

#    if(SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE)
#      add_executable(generate-int8-scale-table generate-int8-scale-table.cc)
#      target_link_libraries(generate-int8-scale-table sherpa-ncnn-core)
#    endif()
  endif()
endif()
#
#if(SHERPA_NCNN_ENABLE_TEST)
#  add_executable(test-resample ${CMAKE_SOURCE_DIR}/runtime/core/test-resample.cc)
#  target_link_libraries(test-resample sherpa-ncnn-core)
#  add_executable(test-context-graph ${CMAKE_SOURCE_DIR}/runtime/core/test-context-graph.cc)
#  target_link_libraries(test-context-graph sherpa-ncnn-core)
#endif()
