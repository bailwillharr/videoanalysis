cmake_minimum_required(VERSION 3.25)

project(calibratetext LANGUAGES CXX)

set(SRC_FILES
  "src/main.cpp"
)

add_executable(${PROJECT_NAME}
  ${SRC_FILES}
)

if(WIN32)
    # stop windows.h conflicting with 'std::max'
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
endif()

# This project uses C++20
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(${PROJECT_NAME} PRIVATE opencv_world)

if (WIN32)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${OpenCV_DLL}
    ${OpenCV_FFMPEG_DLL}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
endif()