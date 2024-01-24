cmake_minimum_required(VERSION 3.25)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-App-Build.cmake")

set(THIS ${ArgProjectName})
project(${THIS})

set(PrivateLibs "")
find_package(OpenSSL REQUIRED)

list(APPEND PrivateLibs "Iterator")
list(APPEND PrivateLibs "Memory")
list(APPEND PrivateLibs "Nexus")
list(APPEND PrivateLibs "xvector")
list(APPEND PrivateLibs "xstring")
list(APPEND PrivateLibs "Macros")
list(APPEND PrivateLibs "xmap")

set(PublicLibs "")
list(APPEND PublicLibs "AES")

BuildRadicalSolution(PrivateLibs PublicLibs)
target_link_libraries(${THIS}
    OpenSSL::SSL 
    OpenSSL::Crypto
)