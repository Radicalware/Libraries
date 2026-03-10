cmake_minimum_required(VERSION 3.25)

FindStaticLib("Stash")

include_directories("${CMAKE_SHARED_INSTALLS}/bsoncxx/v_noabi")
include_directories("${CMAKE_SHARED_INSTALLS}/mongocxx/v_noabi")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(bson CONFIG REQUIRED)
find_package(bsoncxx CONFIG REQUIRED)
find_package(mongocxx CONFIG REQUIRED)


list(APPEND LIST_ONE ${LIST_TWO})

set(LibList
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

