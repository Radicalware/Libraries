cmake_minimum_required(VERSION 3.17)

FindStaticLib("Stash")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(MongoLibs "")
if(${debug})
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/debug/lib/bsoncxx.lib")
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/debug/lib/mongocxx.lib")
else()
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/lib/bsoncxx.lib")
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/lib/mongocxx.lib")
endif()

link_libraries(${LIB}
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MongoLibs}
)
