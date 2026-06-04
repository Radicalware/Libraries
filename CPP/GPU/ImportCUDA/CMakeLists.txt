cmake_minimum_required(VERSION 3.25)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")


# ------------------------------------
set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)
find_package(PrepCUDA)
# ------------------------------------

list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Macros")
BuildRadicalProject("STATIC" PrivateLibs "")

ConfigCUDA(${THIS})