# MakeFile for bash_style.h

cmake_minimum_required(VERSION 3.10)


set(CMAKE_MODULE_PATH /usr/share/cmake/Modules)
find_package(re)
find_package(OS)

project (ex_bash_style)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")

target_link_libraries(${PROJECT_NAME}
	PRIVATE
        rad::OS
)
