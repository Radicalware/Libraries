cmake_minimum_required(VERSION 3.25)

set(CUDA_NVCC_COMPILE_DEFINITIONS "@CUDA_NVCC_COMPILE_DEFINITIONS@") 

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

find_package(PrepCUDA REQUIRED)

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)

set(PrivateLibs "")
list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")
list(APPEND PrivateLibs "xstring")
list(APPEND PrivateLibs "Macros")
list(APPEND PrivateLibs "xmap")
list(APPEND PrivateLibs "ImportCUDA")
list(APPEND PrivateLibs "CudaBridge")
list(APPEND PrivateLibs "Stats")

set(PublicLibs "")
list(APPEND PublicLibs "Timer")

BuildRadicalSolution(PrivateLibs PublicLibs)

ConfigCUDA("ImportCUDA")
ConfigCUDA("CudaBridge")
ConfigCUDA("Stats")
ConfigCUDA(${THIS})
