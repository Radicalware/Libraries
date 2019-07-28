# MakeFile for mc.h

cmake_minimum_required(VERSION 3.10)

set(THIS ex_xmap)
project (${THIS})

message("-----------------------------------------------------------------------------------------------------------")
message("THIS:              ${THIS}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("Install Prefix:    ${INSTALL_PREFIX}")
message("INCLUDE_PATH:      ${INCLUDE_PATH}")
message("LIB_BIN_PATH:      ${LIB_BIN_PATH}")
message("FIND_PATH:         ${FIND_PATH}")
message("-----------------------------------------------------------------------------------------------------------")

# -Wfatal-errors # only Clang
# -std=c++17 # only clang
if(${BUILD_TYPE} STREQUAL "Release")
	add_definitions( -O2 )
endif()
SET(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)

set(CMAKE_MODULE_PATH C:/${FIND_PATH})

find_package(xvector)
find_package(xstring)
find_package(xmap)

add_executable(${THIS}
    src/${THIS}.cpp
)

target_link_libraries( ${THIS}
    PRIVATE 
        radical::xvector
        radical::xstring
        radical::xmap
)
