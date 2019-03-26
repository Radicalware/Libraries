
# Finder for ac.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (ac_lib)

set(AC_DIR /opt/Radicalware/Libraries/cpp/code/ac)
find_path(
    AC_INCLUDE_DIR ac_lib.h
    PATH_SUFFIXES include PATHS
    ${AC_DIR}
)


set(POTENTIAL_AC_LIBS ac aclib libac)

find_library(ac_lib
  NAMES ${POTENTIAL_AC_LIBS}
    PATH_SUFFIXES lib
  PATHS
    ${AC_DIR}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${AC_DIR}/src/ac.cpp)
add_library(gen::ac ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${AC_DIR}/include
)
