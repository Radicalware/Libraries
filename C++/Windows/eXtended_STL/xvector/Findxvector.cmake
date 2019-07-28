cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB xvector)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(XVECTOR_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    STATIC
        ${XVECTOR_DIR}/include/${LIB}.h
        ${XVECTOR_DIR}/src/${LIB}.cpp

        ${XVECTOR_DIR}/include/const_${LIB}.h
        ${XVECTOR_DIR}/src/const_${LIB}.cpp

        ${XVECTOR_DIR}/include/val_${LIB}.h
        ${XVECTOR_DIR}/src/val_${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${XVECTOR_DIR}/include
)
