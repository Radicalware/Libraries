
# Finder for OS.h
cmake_minimum_required(VERSION 3.10)

# Set the project name
project (OS)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

set(OS_DIR /opt/Radicalware/Libraries/cpp/code/OS)

add_library(${PROJECT_NAME} SHARED 
        
    ${OS_DIR}/src/OS.cpp
    ${OS_DIR}/src/support_os/Dir_Type.cpp
    ${OS_DIR}/src/support_os/File_Names.cpp
)

add_library(rad::OS ALIAS ${PROJECT_NAME})


target_link_libraries(${PROJECT_NAME}
    rad::re
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${OS_DIR}/include
    ${OS_DIR}/include/support_os
)
