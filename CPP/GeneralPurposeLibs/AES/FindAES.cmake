cmake_minimum_required(VERSION 3.25)


find_package(OpenSSL REQUIRED)

FindDynamicLib("AES")

link_libraries(${LIB}
    ${PreStaticLibLst}

    OpenSSL::SSL 
    OpenSSL::Crypto
)