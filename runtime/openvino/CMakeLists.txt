include_directories(${CMAKE_SOURCE_DIR})

find_package(OpenVINO REQUIRED)
if(OpenVINO_FOUND)
    message("Found OpenVINO")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(sherpa_openvino_core_srcs
  ${CMAKE_SOURCE_DIR}/runtime/core/context-graph.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/endpoint.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/file-utils.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/hypothesis.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/resample.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/symbol-table.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/wave-reader.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/wave-writer.cc
  ${CMAKE_SOURCE_DIR}/runtime/core/features.cc
  model.cc
  zipformer-model.cc
  stream.cc
  recognizer.cc
  decoder.cc
  greedy-search-decoder.cc
  modified-beam-search-decoder.cc
)

# Always static build so that there is a single shared library: sherpa-openvino-c-api.dll
add_library(sherpa-openvino-core STATIC ${sherpa_openvino_core_srcs})

target_link_libraries(sherpa-openvino-core PUBLIC kaldi-native-fbank-core openvino::runtime)

if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-openvino-core DESTINATION ..)
else()
  install(TARGETS sherpa-openvino-core DESTINATION lib)
endif()

add_subdirectory(c-api)

if(NOT SHERPA_ONNX_ENABLE_PYTHON)
  if(SHERPA_DEPLOY_ENABLE_OPENVINO)
    add_executable(sherpa-openvino sherpa-openvino.cc)
    target_link_libraries(sherpa-openvino PRIVATE sherpa-openvino-core)
    install(TARGETS sherpa-openvino DESTINATION bin)

    if(SHERPA_ONNX_ENABLE_PORTAUDIO)
      add_executable(sherpa-openvino-microphone
        sherpa-openvino-microphone.cc
        ${CMAKE_SOURCE_DIR}/runtime/core/microphone.cc
      )

      set(PA_LIB portaudio_static)

      target_link_libraries(sherpa-openvino-microphone PRIVATE ${PA_LIB} sherpa-openvino-c-api)

      install(TARGETS sherpa-openvino-microphone DESTINATION bin)
    endif()

  endif()
endif()
