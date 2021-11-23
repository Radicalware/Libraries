cmake_minimum_required(VERSION 3.17)

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()
include("${RADICAL_PATH}/Radical-Lib-Build.cmake")

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

BuildRadicalProject("STATIC" PrivateLibs "")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(bsoncxx CONFIG REQUIRED)
find_package(mongocxx CONFIG REQUIRED)

target_link_libraries(${THIS}
    re2::re2

    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal 
    cpprestsdk::cpprestsdk_boost_internal 
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    mongo::bsoncxx_shared
    mongo::mongocxx_shared
)
