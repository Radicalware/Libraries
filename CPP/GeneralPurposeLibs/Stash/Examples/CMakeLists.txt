cmake_minimum_required(VERSION 3.17)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS})

set(PrivateLibs "")
find_package(re2 CONFIG REQUIRED)

list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")
list(APPEND PrivateLibs "xstring")
list(APPEND PrivateLibs "Macros")
list(APPEND PrivateLibs "xmap")
list(APPEND PrivateLibs "JSON")
list(APPEND PrivateLibs "Stash")

BuildRadicalSolution(PrivateLibs "")
link_libraries(${THIS} re2::re2)