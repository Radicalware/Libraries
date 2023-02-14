cmake_minimum_required(VERSION 3.25)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")


# ------------------------------------
set(CUDA_VERSION "12.0")
set(CUDA_GPU 61)
find_package(PrepCUDA)

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)
# ------------------------------------

list(APPEND PrivateLibs "Memory")
BuildRadicalProject("STATIC" PrivateLibs "")

ConfigCUDA(${THIS})