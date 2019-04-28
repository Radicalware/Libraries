# MakeFile for OS.h

cmake_minimum_required(VERSION 3.10)

project (OS)

if(EXISTS "${PROJECT_SOURCE_DIR}/FindOS.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! FindOS.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/Modules")

file(REMOVE ${CMAKE_FIND_PATH}/FindOS.cmake)

add_library(${PROJECT_NAME} SHARED src/OS.cpp)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/support_os
)

set(SOURCES 
  Dir_Type.cpp 
  File_Names.cpp
  OS.cpp 
  ${PROJECT_SOURCE_DIR}/include/support_os/Dir_Type.h
  ${PROJECT_SOURCE_DIR}/include/support_os/File_Names.h
  ${PROJECT_SOURCE_DIR}/include/OS.h
)


# ---------------------------------------------------------------------------

# Binary File
install (  
    TARGETS         ${PROJECT_NAME}
    DESTINATION     bin
)

# OS files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/OS
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/OS
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           FindOS.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
