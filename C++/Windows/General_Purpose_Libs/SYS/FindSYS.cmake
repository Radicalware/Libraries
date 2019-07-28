cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB SYS)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(SYS_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    SHARED 
        ${SYS_DIR}/src/${LIB}.cpp
        ${SYS_DIR}/include/${LIB}.h
)
add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        ${XMAP_DIR}/include
        
        ${SYS_DIR}/include
)

target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)
