
# Finder for SYS.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (SYS_lib)


set(SYS_dir /opt/Radicalware/Libraries/cpp/code/SYS)
find_path(
    SYS_include_dir SYS_lib.h
    PATH_SUFFIXES include PATHS
    ${SYS_dir}
)

set(potential_SYS_libs SYS SYS_lib SYSlib libSYS)

find_library(SYS_lib
  NAMES ${potential_SYS_libs}
    PATH_SUFFIXES lib
  PATHS
    ${SYS_dir}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${SYS_dir}/src/SYS.cpp)
add_library(gen::SYS ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${SYS_dir}/include
)


