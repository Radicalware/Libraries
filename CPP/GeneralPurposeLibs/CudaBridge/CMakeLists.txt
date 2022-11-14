cmake_minimum_required(VERSION 3.17)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)

set(CUDA_VERSION "11.8")
find_package(PrepCUDA)
set(CMAKE_CUDA_STANDARD 17)

list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "Iterator")
BuildRadicalProject("STATIC" PrivateLibs "")

