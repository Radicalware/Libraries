
# Finder for cc.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (cc_lib)


set(cc_dir /opt/Radicalware/Libraries/cpp/code/cc)
find_path(
    cc_include_dir cc_lib.h
    PATH_SUFFIXES include PATHS
    ${cc_dir}
)

set(potential_cc_libs cc cc_lib cclib libcc)

find_library(cc_lib
  NAMES ${potential_cc_libs}
    PATH_SUFFIXES lib
  PATHS
    ${cc_dir}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${cc_dir}/src/cc.cpp)
add_library(gen::cc ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${cc_dir}/include
)


