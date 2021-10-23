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

set(MONGO_LIBS "")
if(${debug})
    list(APPEND ${MONGO_LIBS}
        "${VCPKG_DEBUG_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_DEBUG_LIB_DIR}/mongocxx.lib"
    )
else()
    list(APPEND ${MONGO_LIBS}
        "${VCPKG_RELEASE_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_RELEASE_LIB_DIR}/mongocxx.lib"
    )
endif()

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(${THIS}
    re2::re2

    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MONGO_LIBS}
)
