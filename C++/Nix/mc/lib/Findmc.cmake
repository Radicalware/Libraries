# Finder for mc.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (mc_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(mc_DIR /opt/Radicalware/Libraries/cpp/code/mc)

add_library(${PROJECT_NAME} SHARED ${mc_DIR}/src/mc.cpp)
add_library(rad::mc ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${mc_DIR}/include
)
