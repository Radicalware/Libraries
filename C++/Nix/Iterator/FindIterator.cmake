# Finder for Iterator.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (Iterator_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(Iterator_DIR /opt/Radicalware/Libraries/cpp/code/Iterator)

add_library(${PROJECT_NAME} SHARED ${Iterator_DIR}/src/Iterator.cpp)
add_library(rad::Iterator ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${Iterator_DIR}/include
)
