
# MakeFile for ac.h >> "array control"

cmake_minimum_required(VERSION 3.10)

project (ex_ac_sln)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)
set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(ac)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Add an executable with the above sources
add_executable(ex_ac
    ex_ac.cpp   
)

# link the new hello_library target with the hello_binary target
target_link_libraries( ex_ac
    PRIVATE 
        rad::ac
)
