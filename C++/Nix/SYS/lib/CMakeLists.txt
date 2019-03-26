
# MakeFile for SYS.h

cmake_minimum_required(VERSION 3.10)

project (SYS)

if(EXISTS "${PROJECT_SOURCE_DIR}/FindSYS.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! FindSYS.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/FindSYS.cmake)

add_library(${PROJECT_NAME} SHARED src/SYS.cpp)
add_library(gen::SYS ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------

# Binary File
install (  
    TARGETS         ${PROJECT_NAME}
    DESTINATION     bin
)

# SYS files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/SYS
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/SYS
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           FindSYS.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
