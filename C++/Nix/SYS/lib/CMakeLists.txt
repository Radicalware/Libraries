
# MakeFile for SYS.h

cmake_minimum_required(VERSION 3.10)

project (SYS)


execute_process (COMMAND bash -c "rm -f /usr/share/cmake/Modules/Findac.cmake")
execute_process (
    COMMAND bash -c "cp ../FindSYS.cmake /usr/share/cmake/Modules/"
    OUTPUT_VARIABLE outVar
)

if(EXISTS "/usr/share/cmake/Modules/FindSYS.cmake")
    message(">> Finder Coppied!!")
else()    
    message("\n\n!!!! FindSYS.cmake Failed to Copy !!!!\n\n")
    return()
endif()

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

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
