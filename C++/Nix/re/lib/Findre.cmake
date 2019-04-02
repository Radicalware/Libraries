
# Finder for re.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (re_lib)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(RE_DIR /opt/Radicalware/Libraries/cpp/code/re)

add_library(${PROJECT_NAME} SHARED ${RE_DIR}/src/re.cpp)
add_library(rad::re ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${RE_DIR}/include
)


