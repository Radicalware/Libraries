cmake_minimum_required(VERSION 3.25)

find_package(OpenSSL REQUIRED)

FindDynamicLib("AES")
target_link_libraries(${LIB} PRIVATE
    ${PreStaticLibLst}
    OpenSSL::SSL 
    OpenSSL::Crypto
)