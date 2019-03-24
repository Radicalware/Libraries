# Finder for mc.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (mc_lib)


set(mc_dir /opt/Radicalware/Libraries/cpp/code/mc)
find_path(
    mc_include_dir mc_lib.h
    PATH_SUFFIXES include PATHS
    ${mc_dir}
)


set(potential_mc_libs mc mc_lib mclib libmc)

find_library(mc_lib
  NAMES ${potential_mc_libs}
    PATH_SUFFIXES lib
  PATHS
    ${mc_dir}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${mc_dir}/src/mc.cpp)
add_library(gen::mc ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${mc_dir}/include
)


