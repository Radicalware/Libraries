cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB cc)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(Timer_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    SHARED
        ${Timer_DIR}/include/${LIB}.h
        ${Timer_DIR}/src/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${Timer_DIR}/include
)
