cmake_minimum_required(VERSION 3.10)

# Set the project name
set(LIB OS)

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

set(OS_DIR ${INSTALL_PREFIX}/code/${LIB})

add_library(${LIB} 
    SHARED  
        ${OS_DIR}/include/${LIB}.h
        ${OS_DIR}/src/${LIB}.cpp

        ${OS_DIR}/include/dir_support/Dir_Type.h
        ${OS_DIR}/src/dir_support/Dir_Type.cpp

        ${OS_DIR}/include/dir_support/File_Names.h
        ${OS_DIR}/src/dir_support/File_Names.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        ${XMAP_DIR}/include

        ${OS_DIR}/include
        ${OS_DIR}/include/support_dir
)

target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)
