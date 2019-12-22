
include("${RADICAL_PATH}/Radical-Static-Vars.cmake")
include("${RADICAL_PATH}/Radical-Header.cmake")

# -------------------------- CONFIGURATION ------------------------------------------------------

set(INSTALL_PREFIX "${INSTALL_PREFIX}/Applications")

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Solution)
set(INC       ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INC}) 
set(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS ON)

if(WIN32)
    SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/bin")
    SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/bin")

    SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/lib")
    SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/lib")

    SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/lib")
    SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/lib")
else()
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${BUILD_TYPE}")
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/bin")
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/lib")
endif() 

# -------------------------- CONFIGURATION ------------------------------------------------------
