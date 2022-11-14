cmake_minimum_required(VERSION 3.17)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)
set(CUDA_VERSION "11.8")
set(CMAKE_CUDA_STANDARD 17)


set(PrivateLibs "")
find_package(PrepCUDA REQUIRED)
find_package(re2)
list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "CudaBridge")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")

BuildRadicalSolution(PrivateLibs "")
link_libraries(${THIS} re2::re2)