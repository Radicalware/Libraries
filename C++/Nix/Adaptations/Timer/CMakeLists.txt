
# MakeFile for Timer.h

cmake_minimum_required(VERSION 3.10)

project (Timer)

if(EXISTS "${PROJECT_SOURCE_DIR}/FindTimer.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! FindTimer.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/FindTimer.cmake)

add_library(${PROJECT_NAME} SHARED src/Timer.cpp)
add_library(gen::Timer ALIAS ${PROJECT_NAME})

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

# Timer files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/Timer
)

install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/Timer
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           FindTimer.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
