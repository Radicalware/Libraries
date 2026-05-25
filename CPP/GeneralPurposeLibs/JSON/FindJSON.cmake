cmake_minimum_required(VERSION 3.25)

# because VCPKG uses an older function
add_definitions(-D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)

FindStaticLib("JSON")

set(BSON_INCLUDE  "${VCPKG_INCLUDE}/bsoncxx/v_noabi")
set(MONGO_INCLUDE "${VCPKG_INCLUDE}/mongocxx/v_noabi")
#set(MONGO_PACKAGE "${VCPKG_PACKAGES}/mongo-cxx-driver_x64-windows/include/bsoncxx/v_noabi")

list(APPEND InstalledIncludeDirs "${BSON_INCLUDE}")
include_directories("${BSON_INCLUDE}")

list(APPEND InstalledIncludeDirs "${MONGO_INCLUDE}")
include_directories("${MONGO_INCLUDE}")
#include_directories("${MONGO_PACKAGE}")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(bson CONFIG REQUIRED)
find_package(bsoncxx CONFIG REQUIRED)
find_package(mongocxx CONFIG REQUIRED)

set(UsedVcpkgLibs
    cpprestsdk::cpprest 
    cpprestsdk::cpprestsdk_zlib_internal 
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    mongo::bsoncxx_shared
    mongo::mongocxx_shared
)

list(REMOVE_DUPLICATES UsedVcpkgLibs)
set(COMBINED_LIST ${PreStaticLibLst} ${UsedVcpkgLibs})
list(REMOVE_DUPLICATES COMBINED_LIST)
link_libraries(${LIB}
    ${COMBINED_LIST}
)
list(APPEND UsedVcpkgLibs ${UsedVcpkgLibs})
