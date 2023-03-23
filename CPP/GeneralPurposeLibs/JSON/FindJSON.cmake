cmake_minimum_required(VERSION 3.25)

FindStaticLib("JSON")

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(bson-1.0 CONFIG REQUIRED)
find_package(bsoncxx CONFIG REQUIRED)
find_package(mongocxx CONFIG REQUIRED)

link_libraries(${LIB}
${PreStaticLibLst}
    
    cpprestsdk::cpprest 
    cpprestsdk::cpprestsdk_zlib_internal 
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    mongo::bsoncxx_shared
    mongo::bson_shared
    mongo::mongocxx_shared
)