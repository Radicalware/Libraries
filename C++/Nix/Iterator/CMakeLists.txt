
# CMakeFile for Iterator.h

cmake_minimum_required(VERSION 3.10)

project (Iterator)

if(EXISTS "${PROJECT_SOURCE_DIR}/FindIterator.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! FindIterator.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/FindIterator.cmake)

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED src/Iterator.cpp)
add_library(rad::Iterator ALIAS ${PROJECT_NAME})

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

# Header files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/Iterator
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/Iterator
)

# Config
install (   
    FILES           FindIterator.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)

