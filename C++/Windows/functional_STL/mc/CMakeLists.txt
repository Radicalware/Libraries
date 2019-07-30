
cmake_minimum_required(VERSION 3.10)

set(THIS mc)
project (${THIS})

if(EXISTS "${PROJECT_SOURCE_DIR}/Find${PROJECT_NAME}.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("/n/n!!!! Find${PROJECT_NAME}.cmake NOT FOUND !!!!/n/n")
    return()
endif()

message("-----------------------------------------------------------------------------------------------------------")
message("THIS:              ${THIS}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("INSTALL_PREFIX:    ${INSTALL_PREFIX}")
message("INCLUDE_PATH:      ${INCLUDE_PATH}")
message("LIB_BIN_PATH:      ${LIB_BIN_PATH}")
message("FIND_PATH:         ${FIND_PATH}")
message("-----------------------------------------------------------------------------------------------------------")

if("${BUILD_TYPE}" STREQUAL "Release")
	message("Buidling with -O2 ${BUILD_TYPE}")
	add_definitions( -O2 )
endif()

file(REMOVE ${FIND_PATH}/Find${THIS}.cmake)

SET(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)

add_library(${THIS} STATIC src/${THIS}.cpp)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${INCLUDE_PATH}
)

# Required Libs -------------------------------------------------------------

# ---------------------------------------------------------------------------

# Binary File
install (  
    TARGETS         ${THIS}
    DESTINATION     bin/${BUILD_TYPE}
)

# Header files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     /
)

# Code files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/${THIS}
)
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/${THIS}
)

# Config
install (   
    FILES           Find${THIS}.cmake 
    DESTINATION     C:/${FIND_PATH}
)