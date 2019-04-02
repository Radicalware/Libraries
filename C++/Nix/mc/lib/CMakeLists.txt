
# MakeFile for mc.h

cmake_minimum_required(VERSION 3.10)

project (mc)

if(EXISTS "${PROJECT_SOURCE_DIR}/Findmc.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! Findmc.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/Findmc.cmake)

add_library(${PROJECT_NAME} SHARED src/mc.cpp)
add_library(rad::mc ALIAS ${PROJECT_NAME})

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

# mc files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/mc
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/mc
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           Findmc.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
