# Finder for Timer.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (Timer_lib)


set(Timer_dir /opt/Radicalware/Libraries/cpp/code/Timer)
find_path(
    Timer_include_dir Timer_lib.h
    PATH_SUFFIXES include PATHS
    ${Timer_dir}
)


set(potential_Timer_libs Timer Timer_lib Timerlib libTimer)

find_library(Timer_lib
  NAMES ${potential_Timer_libs}
    PATH_SUFFIXES lib
  PATHS
    ${Timer_dir}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${Timer_dir}/src/Timer.cpp)
add_library(gen::Timer ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    INTERFACE
        ${Timer_dir}/include
)


