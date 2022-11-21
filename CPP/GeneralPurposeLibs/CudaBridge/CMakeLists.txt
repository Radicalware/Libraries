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
set(CUDA_GPU 61)
find_package(PrepCUDA)

list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
BuildRadicalProject("STATIC" PrivateLibs "")

if(${debug})
    target_compile_options(${THIS} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           --generate-line-info
                           --use_fast_math
                           --relocatable-device-code=true
                           >)
else()
    target_compile_options(${THIS} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           --use_fast_math
                           --relocatable-device-code=true
                           >)
endif()