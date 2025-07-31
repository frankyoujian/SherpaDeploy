
include(FetchContent)

set(itt_URL  "https://github.com/intel/ittapi/archive/refs/tags/v3.26.1.zip")
set(itt_HASH "SHA256=7083D016D0BAA633714C167B5101D0A301508F29D36E768BAFF1D5517968FA32")

# If you don't have access to the Internet, please download it to your
# local drive and modify the following line according to your needs.
set(possible_file_locations
    $ENV{HOME}/Downloads/v3.26.1.zip
    $ENV{HOME}/asr/v3.26.1.zip
    ${PROJECT_SOURCE_DIR}/v3.26.1.zip
    ${PROJECT_BINARY_DIR}/v3.26.1.zip
    /tmp/v3.26.1.zip
)

foreach(f IN LISTS possible_file_locations)
if(EXISTS ${f})
    set(itt_URL  "${f}")
    file(TO_CMAKE_PATH "${itt_URL}" itt_URL)
    break()
endif()
endforeach()


FetchContent_Declare(itt
    URL
    ${itt_URL}
    URL_HASH          ${itt_HASH}
)

FetchContent_GetProperties(itt)
if(NOT itt_POPULATED)
    message(STATUS "Downloading ittapi from ${itt_URL}")
    FetchContent_Populate(itt)
endif()
message(STATUS "ittapi is downloaded to ${itt_SOURCE_DIR}")

add_subdirectory(${itt_SOURCE_DIR} ${itt_BINARY_DIR} EXCLUDE_FROM_ALL)

install(TARGETS ittnotify DESTINATION ..)