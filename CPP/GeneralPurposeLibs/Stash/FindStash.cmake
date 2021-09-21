cmake_minimum_required(VERSION 3.17)

FindStaticLib("Stash")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(MongoLibs "")
if(${debug})
    list(APPEND ${MongoLibs}
        "${VCPKG_DEBUG_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_DEBUG_LIB_DIR}/mongocxx.lib"
    )
else()
    list(APPEND ${MongoLibs}
        "${VCPKG_RELEASE_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_RELEASE_LIB_DIR}/mongocxx.lib"
    )
endif()

link_libraries(${LIB}
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MongoLibs}
)
