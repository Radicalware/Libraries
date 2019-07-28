
cmake_minimum_required(VERSION 3.10)

set(THIS ex_OS)
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

find_package(xvector)
find_package(xstring)
find_package(xmap)
find_package(OS)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${THIS}
    src/${THIS}.cpp

	include/ex_bash_style.h
    include/ex_file_managment.h
    include/ex_open_n_delete.h
)

target_link_libraries( ${THIS}
    PRIVATE 
        radical::xvector
        radical::xstring
        radical::xmap
        radical::OS
)
