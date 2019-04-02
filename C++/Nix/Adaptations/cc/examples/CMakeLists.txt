
# MakeFile for cc.h

cmake_minimum_required(VERSION 3.10)

project (ex_cc_sln)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)
set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(cc)

# Add an executable with the above sources
add_executable(ex_cc
    ex_cc.cpp   
)

# link the new hello_library target with the hello_binary target
target_link_libraries( ex_cc
    PRIVATE 
		rad::cc
)
