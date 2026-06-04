cmake_minimum_required(VERSION 4.3.3)

set(CMAKE_CXX_IGNORE_EXTENSIONS "${CMAKE_CXX_IGNORE_EXTENSIONS};txt;rc")
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS "${CMAKE_CXX_SOURCE_FILE_EXTENSIONS};cuh;cu")
# --------------------------------------------------------------------------------------------
if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")

set(THIS ${ArgProjectName})

# ------------------------------------
#project(${THIS})
# ------------------------------------
project(${THIS} LANGUAGES CXX CUDA)
find_package(PrepCUDA)
# ------------------------------------

set(PrivateLibs "")
list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")
list(APPEND PrivateLibs "xstring")
list(APPEND PrivateLibs "Macros")
list(APPEND PrivateLibs "ImportCUDA")
list(APPEND PrivateLibs "CudaBridge")

BuildRadicalProject("STATIC" PrivateLibs "")

ConfigCUDA("ImportCUDA")
ConfigCUDA("CudaBridge")
ConfigCUDA(${THIS})

