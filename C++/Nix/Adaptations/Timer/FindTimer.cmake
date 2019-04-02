# Finder for Timer.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (Timer_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(Timer_DIR /opt/Radicalware/Libraries/cpp/code/Timer)

add_library(${PROJECT_NAME} SHARED ${Timer_DIR}/src/Timer.cpp)
add_library(rad::Timer ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${Timer_DIR}/include
)
