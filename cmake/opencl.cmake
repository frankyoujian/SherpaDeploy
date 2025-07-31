
include(FetchContent)

set(opencl_URL  "https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2024.10.24/OpenCL-SDK-v2024.10.24-Win-x64.zip")
set(opencl_HASH "SHA256=F849524BE3691B0E601B3DC50E2AAA458AF0B4E3357B5D17525E8A3ECB44F3F6")

# If you don't have access to the Internet, please download it to your
# local drive and modify the following line according to your needs.
set(possible_file_locations
    $ENV{HOME}/Downloads/OpenCL-SDK-v2024.10.24-Win-x64.zip
    $ENV{HOME}/asr/OpenCL-SDK-v2024.10.24-Win-x64.zip
    ${PROJECT_SOURCE_DIR}/OpenCL-SDK-v2024.10.24-Win-x64.zip
    ${PROJECT_BINARY_DIR}/OpenCL-SDK-v2024.10.24-Win-x64.zip
    /tmp/OpenCL-SDK-v2024.10.24-Win-x64.zip
)

foreach(f IN LISTS possible_file_locations)
if(EXISTS ${f})
    set(opencl_URL  "${f}")
    file(TO_CMAKE_PATH "${opencl_URL}" opencl_URL)
    break()
endif()
endforeach()

FetchContent_Declare(opencl
    URL
    ${opencl_URL}
    URL_HASH          ${opencl_HASH}
)

FetchContent_GetProperties(opencl)
if(NOT opencl_POPULATED)
    message(STATUS "Downloading opencl from ${opencl_URL}")
    FetchContent_Populate(opencl)
endif()
message(STATUS "opencl is downloaded to ${opencl_SOURCE_DIR}")

find_library(opencl opencl
  PATHS
  "${opencl_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
include_directories(${opencl_SOURCE_DIR}/include)
message(STATUS "opencl_lib: ${opencl}")

file(COPY ${opencl_SOURCE_DIR}/bin/OpenCL.dll
  DESTINATION
    ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}
)