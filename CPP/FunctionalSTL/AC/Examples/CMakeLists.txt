﻿cmake_minimum_required(VERSION 3.25)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS})

set(PrivateLibs "")
list(APPEND PrivateLibs "AC")

BuildRadicalSolution(PrivateLibs "")