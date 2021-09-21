set(IsLib ON)
set(IsApp OFF)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

include("${RADICAL_PATH}/Radical-Includes.cmake")
PrintBanner()

# -------------------------- CONFIGURATION ------------------------------------

set(INSTALL_PREFIX "${INSTALL_PREFIX}/Libraries")

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Project)
set(INC       ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

set(cmake_configuration_types ${build_type} cache string "" force)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INC}) 
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FindIncludeDirs(InstalledIncludeDirs "${INSTALL_DIR}")

# -------------------------- CONFIGURATION ------------------------------------
