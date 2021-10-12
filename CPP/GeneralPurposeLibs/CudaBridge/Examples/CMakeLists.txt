cmake_minimum_required(VERSION 3.17)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
set(CUDA_VERSION "11.4")
find_package(PrepCUDA REQUIRED)
project(${THIS} LANGUAGES CXX CUDA)

set(PrivateLibs "")
list(APPEND PrivateLibs "CudaBridge")

BuildRadicalSolution(PrivateLibs "")