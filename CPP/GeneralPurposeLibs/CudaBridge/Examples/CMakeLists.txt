cmake_minimum_required(VERSION 3.24)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)

set(CUDA_VERSION "11.8")
set(CUDA_GPU 61)
find_package(PrepCUDA REQUIRED)

set(PrivateLibs "")
list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")
list(APPEND PrivateLibs "xstring")
list(APPEND PrivateLibs "Macros")
list(APPEND PrivateLibs "CudaBridge")

set(PublicLibs "")
list(APPEND PublicLibs "Timer")
BuildRadicalSolution(PrivateLibs PublicLibs)

FinishConfiguringCUDA()

