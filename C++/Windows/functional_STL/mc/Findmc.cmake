cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB mc)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(MC_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    STATIC
        ${MC_DIR}/include/${LIB}.h
        ${MC_DIR}/src/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${MC_DIR}/include
)
