cmake_minimum_required(VERSION 3.17)

find_package(re2)
FindStaticLib("xvector")
link_libraries(${LIB} re2::re2)