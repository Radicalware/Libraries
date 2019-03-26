
# MakeFile for cc.h

cmake_minimum_required(VERSION 3.10)

project (cc)

if(EXISTS "${PROJECT_SOURCE_DIR}/Findcc.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! Findcc.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/Findcc.cmake)

add_library(${PROJECT_NAME} SHARED src/cc.cpp)
add_library(gen::cc ALIAS ${PROJECT_NAME})

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
    DESTINATION     code/cc
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/cc
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           Findcc.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
