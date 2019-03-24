
# MakeFile for mc.h

cmake_minimum_required(VERSION 3.10)

project (ex_mc_sln)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)
set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(mc)
find_package(ac)

# Add an executable with the above sources
add_executable(ex_mc
    ex_mc.cpp   
)

# link the new hello_library target with the hello_binary target
target_link_libraries( ex_mc
    PRIVATE 
        gen::ac
		gen::mc
)
