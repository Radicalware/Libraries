cmake_minimum_required(VERSION 3.25)

# because VCPKG uses an older function
add_definitions(-D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)

FindStaticLib("JSON")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(bson-1.0 CONFIG REQUIRED)
find_package(bsoncxx CONFIG REQUIRED)
find_package(mongocxx CONFIG REQUIRED)

set(BSON_INCLUDE "${VCPKG_INCLUDE}/bsoncxx/v_noabi")
list(APPEND InstalledIncludeDirs "${BSON_INCLUDE}")
include_directories("${BSON_INCLUDE}")

set(MONGO_INCLUDE "${VCPKG_INCLUDE}/mongocxx/v_noabi")
list(APPEND InstalledIncludeDirs "${MONGO_INCLUDE}")
include_directories("${MONGO_INCLUDE}")

set(UsedVcpkgLibs
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_boost_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    mongo::bsoncxx_shared
    mongo::mongocxx_shared
    mongo::bson_shared
)

list(REMOVE_DUPLICATES UsedVcpkgLibs)
set(COMBINED_LIST ${PreStaticLibLst} ${UsedVcpkgLibs})
list(REMOVE_DUPLICATES COMBINED_LIST)
link_libraries(${LIB}
    ${COMBINED_LIST}
)
list(APPEND UsedVcpkgLibs ${UsedVcpkgLibs})
