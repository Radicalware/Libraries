cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB ac)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(AC_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    STATIC
        ${AC_DIR}/include/${LIB}.h
        ${AC_DIR}/src/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${AC_DIR}/include
)
