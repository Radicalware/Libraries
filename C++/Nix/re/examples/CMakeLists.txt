
# MakeFile for re.h >> regex

cmake_minimum_required(VERSION 3.10)

project (ex_re_sln)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)
set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(re)
find_package(ac)

# Add an executable with the above sources
add_executable(ex_re
    ex_re.cpp   
)

# link the new hello_library target with the hello_binary target
target_link_libraries( ex_re
    PRIVATE 
        gen::re
        gen::ac
)
