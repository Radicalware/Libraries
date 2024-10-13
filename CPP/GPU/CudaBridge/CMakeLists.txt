cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0128 NEW)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")

find_package(PrepCUDA)

set(THIS ${ArgProjectName})
project(${THIS} LANGUAGES CXX CUDA)

list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "ImportCUDA")
BuildRadicalProject("STATIC" PrivateLibs "")

ConfigCUDA("ImportCUDA")
ConfigCUDA(${THIS})