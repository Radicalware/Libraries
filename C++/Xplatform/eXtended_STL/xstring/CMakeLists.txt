﻿
cmake_minimum_required(VERSION 3.10)

set(THIS xstring)
project (${THIS})

message("")
message("--------------------------------------------------------------------------------")
message("THIS:              ${THIS}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("INSTALL_PREFIX:    ${INSTALL_PREFIX}")
message("EXT_INCLUDE_PATH:  ${EXT_INCLUDE_PATH}")
message("EXT_BIN_PATH:      ${EXT_BIN_PATH}")
message("MODULE_PATH:       ${MODULE_PATH}")
message("--------------------------------------------------------------------------------")

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    set(PF "")
    set(SF ".lib")

    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling with -O2 ${BUILD_TYPE}")
        add_definitions( "-O2" )
    endif()
else() # NIX
    set(PF "lib")
    set(SF ".a")
    set(LINUX_ARGS "-std=c++17 -Wfatal-errors -finput-charset=UTF-8 -fPIC")

    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling with -O2 ${BUILD_TYPE}")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/lib)
set(INCLUDE   ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

set(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INCLUDE}) 

if(EXISTS "${PROJECT_SOURCE_DIR}/Find${THIS}.cmake")
    file(REMOVE ${MODULE_PATH}/Find${THIS}.cmake)
else()
    message(FATAL_ERROR "/n/n!!!! Find${THIS}.cmake NOT FOUND !!!!/n/n")
endif()
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${THIS} STATIC ${SRC}/${THIS}.cpp)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_INCLUDE_PATH}
        ${INCLUDE}
)
# Required Libs ---------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/${BUILD_TYPE}/${PF}xvector${SF})
# -------------------------- BUILD --------------------------------------------
# -------------------------- INSTALL ------------------------------------------

# Binary File
install (  
    TARGETS             ${THIS}
    DESTINATION         bin/${BUILD_TYPE}
)

# Include file
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         .
)

# Header/Src Files
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         code/${THIS}
)
install (   
    DIRECTORY           ${BUILD_DIR}/src 
    DESTINATION         code/${THIS}
)

# CMake Module
if(UNIX)
    install (   
        FILES           Find${THIS}.cmake 
        DESTINATION     /${PART_MODULE_PATH}
    ) 
else() 
    install (   
        FILES           Find${THIS}.cmake 
        DESTINATION     C:/${PART_MODULE_PATH}
    )
endif()

# -------------------------- INSTALL ------------------------------------------
# -------------------------- END ----------------------------------------------