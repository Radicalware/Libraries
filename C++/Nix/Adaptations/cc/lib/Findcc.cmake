# Finder for cc.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (cc_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(cc_DIR /opt/Radicalware/Libraries/cpp/code/cc)

add_library(${PROJECT_NAME} SHARED ${cc_DIR}/src/cc.cpp)
add_library(rad::cc ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${cc_DIR}/include
)
