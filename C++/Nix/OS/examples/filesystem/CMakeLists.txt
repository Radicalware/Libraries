# MakeFile for open_n_delete.h

cmake_minimum_required(VERSION 3.10)


set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(re)
find_package(OS)

project (ex_file_managment)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)
set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

target_link_libraries( ${PROJECT_NAME}
    PRIVATE
        rad::OS
)
