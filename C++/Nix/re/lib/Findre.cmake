
# Finder for re.h

cmake_minimum_required(VERSION 3.10)

# Set the project name
project (re_lib)


set(RE_DIR /opt/Radicalware/Libraries/cpp/code/re)
find_path(
  RE_INCLUDE_DIR re_lib.h
  PATH_SUFFIXES include PATHS
  ${RE_DIR}
)


set(POTENTIAL_RE_LIBS re relib libre)

find_library(re_lib
  NAMES ${POTENTIAL_RE_LIBS}
    PATH_SUFFIXES lib
  PATHS
    ${RE_DIR}
)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED ${RE_DIR}/src/re.cpp)
add_library(gen::re ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${RE_DIR}/include
)


