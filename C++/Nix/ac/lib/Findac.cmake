# Finder for ac.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (ac_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(ac_DIR /opt/Radicalware/Libraries/cpp/code/ac)

add_library(${PROJECT_NAME} SHARED ${ac_DIR}/src/ac.cpp)
add_library(rad::ac ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${ac_DIR}/include
)
