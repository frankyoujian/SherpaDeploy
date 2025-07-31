message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL X64 OR CMAKE_VS_PLATFORM_NAME STREQUAL x64))
  message(FATAL_ERROR "This file is for Windows x64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

set(openvino_URL  "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/windows/openvino_toolkit_windows_2025.2.0.19140.c01cd93e24d_x86_64.zip")
set(openvino_HASH "SHA256=2ff4c46e089d6f64c8d39eb175e9af7c0fea8052c06968b753a55aa9c83b8de9")

FetchContent_Declare(openvino
  URL               ${openvino_URL}
  URL_HASH          ${openvino_HASH}
)

FetchContent_GetProperties(openvino)
if(NOT openvino_POPULATED)
  message(STATUS "Downloading openvino from ${openvino_URL}")
  FetchContent_Populate(openvino)
endif()
message(STATUS "openvino is downloaded to ${openvino_SOURCE_DIR}")

set(OpenVINO_DIR ${openvino_SOURCE_DIR}/runtime/cmake)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(OPENVINO_PACKAGE_VERSION "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}")

if (CMAKE_SYSTEM_NAME MATCHES Windows)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
  set(INSTALL_DESTINATION bin)
else()
  set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".dylib")
  set(INSTALL_DESTINATION lib)
endif()

# set(OPENVINO_TEMP_LIB_DIR ${INSTALL_FILE_TEMP_DIR}/OpenVINO/${INSTALL_DESTINATION})

if (CMAKE_SYSTEM_NAME MATCHES Windows)
  set(TBB_LIB_NAME tbb12)
else()
  set(TBB_LIB_NAME tbb)
  find_package(OpenVINO REQUIRED COMPONENTS Threading)
  list(APPEND VINO_LIB openvino::threading)

  if (CMAKE_SYSTEM_NAME MATCHES Linux)
    set(TBB_DIR ${OPENVINO_ROOT_DIR}/runtime/3rdparty/tbb/lib)
  endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(TBB_LIB_SUFFIX "_debug") 
else()
  set(TBB_LIB_SUFFIX "")
endif()

set(TBB_LIB_NAME_FULL "${TBB_LIB_NAME}${TBB_LIB_SUFFIX}") 

find_library(
  TBB_LIBRARY
  NAMES
    ${TBB_LIB_NAME_FULL} 
  HINTS
    ${openvino_SOURCE_DIR}/runtime/3rdparty/tbb
  PATH_SUFFIXES
    bin
    lib
  NO_DEFAULT_PATH
  REQUIRED
)
list(APPEND OV_LIBS ${TBB_LIBRARY})

if (CMAKE_SYSTEM_NAME MATCHES Linux)
  find_library(
    OPENCL_LIBRARY
    NAMES
      OpenCL
    #REQUIRED
  )

  if (OPENCL_LIBRARY)
    file(
      COPY 
        ${OPENCL_LIBRARY}
      DESTINATION 
        ${OPENVINO_TEMP_LIB_DIR}
      FOLLOW_SYMLINK_CHAIN
    )
  endif()
  
endif()

find_library(
  OPENVINO_LIBRARY
  NAMES
    openvino
    openvinod
  HINTS
    ${openvino_SOURCE_DIR}/runtime
  PATH_SUFFIXES
    lib/intel64 # Linux
    bin/intel64/${CMAKE_BUILD_TYPE} # Windows
    lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
    lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
  REQUIRED
)
list(APPEND OV_LIBS ${OPENVINO_LIBRARY})

find_library(
  OPENVINO_IR_FRONTEND_LIBRARY
  NAMES
    libopenvino_ir_frontend.so.${OPENVINO_LINK_VERSION}
    openvino_ir_frontend.${OPENVINO_LINK_VERSION}
    openvino_ir_frontend
    openvino_ir_frontendd
  HINTS
    ${openvino_SOURCE_DIR}/runtime
  PATH_SUFFIXES
    lib/intel64 # Linux
    bin/intel64/${CMAKE_BUILD_TYPE} # Windows
    lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
    lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
  REQUIRED
)
list(APPEND OV_LIBS ${OPENVINO_IR_FRONTEND_LIBRARY})

find_library(
  OPENVINO_INTEL_CPU_PLUGIN
  NAMES
    openvino_intel_cpu_plugin
    openvino_arm_cpu_plugin
    openvino_intel_cpu_plugind
    openvino_arm_cpu_plugind
  HINTS
    ${openvino_SOURCE_DIR}/runtime
  PATH_SUFFIXES
    lib/intel64 # Linux
    bin/intel64/${CMAKE_BUILD_TYPE} # Windows
    lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
    lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
  REQUIRED
)
list(APPEND OV_LIBS ${OPENVINO_INTEL_CPU_PLUGIN})

if (NOT APPLE)
  find_library(
    OPENVINO_INTEL_GPU_PLUGIN
    NAMES
      openvino_intel_gpu_plugin
      openvino_intel_gpu_plugind
    HINTS
      ${openvino_SOURCE_DIR}/runtime
    PATH_SUFFIXES
      lib/intel64 # Linux
      bin/intel64/${CMAKE_BUILD_TYPE} # Windows
      lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
      lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
    REQUIRED
  )
  list(APPEND OV_LIBS ${OPENVINO_INTEL_GPU_PLUGIN})
endif()

# openvino_auto_plugin is used for Multi-Device Execution.
find_library(
  OPENVINO_AUTO_PLUGIN
  NAMES
    openvino_auto_plugin
    openvino_auto_plugind
  HINTS
    ${openvino_SOURCE_DIR}/runtime
  PATH_SUFFIXES
    lib/intel64 # Linux
    bin/intel64/${CMAKE_BUILD_TYPE} # Windows
    lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
    lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
  REQUIRED
)
list(APPEND OV_LIBS ${OPENVINO_AUTO_PLUGIN})

find_library(
  OPENVINO_AUTO_BATCH_PLUGIN
  NAMES
    openvino_auto_batch_plugin
    openvino_auto_batch_plugind
  HINTS
    ${openvino_SOURCE_DIR}/runtime
  PATH_SUFFIXES
    lib/intel64 # Linux
    bin/intel64/${CMAKE_BUILD_TYPE} # Windows
    lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
    lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
)

if (OPENVINO_AUTO_BATCH_PLUGIN)
  list(APPEND OV_LIBS ${OPENVINO_AUTO_BATCH_PLUGIN})
endif()

# NPU
if (OPENVINO_PACKAGE_VERSION VERSION_GREATER_EQUAL 2024.3)
  find_library(
    OPENVINO_INTEL_NPU_PLUGIN
    NAMES
      openvino_intel_npu_plugin
      openvino_intel_npu_plugind
    HINTS
      ${openvino_SOURCE_DIR}/runtime
    PATH_SUFFIXES
      lib/intel64 # Linux
      bin/intel64/${CMAKE_BUILD_TYPE} # Windows
      lib/intel64/${CMAKE_BUILD_TYPE} # MacOS x86-64
      lib/arm64/${CMAKE_BUILD_TYPE} # MacOS Arm64
  )

  if (OPENVINO_INTEL_NPU_PLUGIN)
    list(APPEND OV_LIBS ${OPENVINO_INTEL_NPU_PLUGIN})
  endif()
endif()

file(
 COPY
   ${OV_LIBS}
 DESTINATION
   ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)

message(STATUS "OV_LIBS files: ${OV_LIBS}")

if(SHERPA_ONNX_ENABLE_PYTHON)
 install(FILES ${OV_LIBS} DESTINATION ..)
else()
 install(FILES ${OV_LIBS} DESTINATION lib)
endif()

install(FILES ${OV_LIBS} DESTINATION bin)
