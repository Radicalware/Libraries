﻿
set(IsApp ON)
set(IsLib OFF)

include("${RADICAL_PATH}/Radical-Includes.cmake")
PrintBanner()

# -------------------------- CONFIGURATION ------------------------------------------------------

set(INSTALL_PREFIX "${INSTALL_PREFIX}/Applications")

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Solution)
set(INC       ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INC}) 
set(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- CONFIGURATION ------------------------------------------------------
