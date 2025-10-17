cmake_minimum_required(VERSION 3.25)

project(videoanalysis)

if (WIN32)
set(OpenCV_DIR "../opencv/build")
endif()
find_package(OpenCV REQUIRED)
if(WIN32)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    get_target_property(OpenCV_DLL opencv_world IMPORTED_LOCATION_DEBUG)
else()
    get_target_property(OpenCV_DLL opencv_world IMPORTED_LOCATION_RELEASE)
endif()
endif()
file(GLOB OpenCV_FFMPEG_DLL "${OpenCV_DIR}/bin/*.dll")

add_subdirectory(simple_decoder)