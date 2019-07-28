
cmake_minimum_required(VERSION 3.10)

SET(THIS ex_cc)
project (${THIS})

message("-----------------------------------------------------------------------------------------------------------")
message("THIS:              ${THIS}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("Install Prefix:    ${INSTALL_PREFIX}")
message("INCLUDE_PATH:      ${INCLUDE_PATH}")
message("LIB_BIN_PATH:      ${LIB_BIN_PATH}")
message("FIND_PATH:         ${FIND_PATH}")
message("-----------------------------------------------------------------------------------------------------------")

if(${BUILD_TYPE} STREQUAL "Release")
	message("Buidling with -O2")
	add_definitions( -O2 )
endif()

SET(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)

set(CMAKE_MODULE_PATH C:/${FIND_PATH})

find_package(cc)

add_executable(${THIS}
    src/${THIS}.cpp
)

target_link_libraries( ${THIS}
    PRIVATE
        radical::cc
)
