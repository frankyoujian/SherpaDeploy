function(download_MNN)
  include(FetchContent)

  # the latest master as of 2025.05.07
  set(MNN_URL  "https://github.com/alibaba/MNN/archive/16bc90912d477111331ab15ac29137f3294f8f69.zip")
  set(MNN_HASH "SHA256=9F6AC690AE7299A56362AA98521822D630693BAB5D0D93D43F99C7CA49A99697")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/MNN-16bc90912d477111331ab15ac29137f3294f8f69.zip
    $ENV{HOME}/asr/MNN-16bc90912d477111331ab15ac29137f3294f8f69.zip
    ${PROJECT_SOURCE_DIR}/MNN-16bc90912d477111331ab15ac29137f3294f8f69.zip
    ${PROJECT_BINARY_DIR}/MNN-16bc90912d477111331ab15ac29137f3294f8f69.zip
    /tmp/MNN-16bc90912d477111331ab15ac29137f3294f8f69.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(MNN_URL  "${f}")
      file(TO_CMAKE_PATH "${MNN_URL}" MNN_URL)
      set(MNN_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(MNN
    URL
      ${MNN_URL}
      ${MNN_URL2}
    URL_HASH  ${MNN_HASH}
  )

  set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)

  set(MNN_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
  # Build MNN-MINI that just supports fixed shape models. This toggle does not support on Windows
  set(MNN_BUILD_MINI OFF CACHE BOOL "" FORCE)

  # backend options
  set(MNN_METAL ${MNN_METAL} CACHE BOOL "" FORCE)
  set(MNN_OPENCL ${MNN_OPENCL} CACHE BOOL "" FORCE)
  set(MNN_OPENGL ${MNN_OPENGL} CACHE BOOL "" FORCE)
  set(MNN_VULKAN ${MNN_VULKAN} CACHE BOOL "" FORCE)
  set(MNN_ARM82 ${MNN_ARM82} CACHE BOOL "" FORCE)
  set(MNN_SUPPORT_FP16_ARMV7 ${MNN_SUPPORT_FP16_ARMV7} CACHE BOOL "" FORCE)
  set(MNN_KLEIDIAI ${MNN_KLEIDIAI} CACHE BOOL "" FORCE)
  set(MNN_ONEDNN ${MNN_ONEDNN} CACHE BOOL "" FORCE)
  set(MNN_AVX2 ${MNN_AVX2} CACHE BOOL "" FORCE)
  set(MNN_AVX512 ${MNN_AVX512} CACHE BOOL "" FORCE)
  set(MNN_CUDA ${MNN_CUDA} CACHE BOOL "" FORCE)
  set(MNN_TENSORRT ${MNN_TENSORRT} CACHE BOOL "" FORCE)
  set(MNN_COREML ${MNN_COREML} CACHE BOOL "" FORCE)
  set(MNN_NNAPI ${MNN_NNAPI} CACHE BOOL "" FORCE)

  FetchContent_GetProperties(MNN)
  if(NOT MNN_POPULATED)
    message(STATUS "Downloading MNN from ${MNN_URL}")
    FetchContent_Populate(MNN)
  endif()
  message(STATUS "MNN is downloaded to ${mnn_SOURCE_DIR}")
  message(STATUS "MNN's binary dir is ${mnn_BINARY_DIR}")

  set_property(GLOBAL APPEND PROPERTY MNN_HEADERS_DIR ${mnn_SOURCE_DIR}/include)
  set_property(GLOBAL APPEND PROPERTY MNN_HEADERS_DIR ${mnn_SOURCE_DIR}/tools/cpp)
  add_subdirectory(${mnn_SOURCE_DIR} ${mnn_BINARY_DIR} EXCLUDE_FROM_ALL)
  if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
    install(TARGETS MNN DESTINATION ..)
  else()
    install(TARGETS MNN DESTINATION lib)
  endif()
endfunction()

download_MNN()
