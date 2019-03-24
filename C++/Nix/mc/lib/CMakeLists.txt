# MakeFile for mc.h

cmake_minimum_required(VERSION 3.10)

project (mc)

execute_process (COMMAND bash -c "rm -f /usr/share/cmake/Modules/Findmc.cmake")
execute_process (
    COMMAND bash -c "cp ../Findmc.cmake /usr/share/cmake/Modules/"
    OUTPUT_VARIABLE outVar
)

if(EXISTS "/usr/share/cmake/Modules/Findmc.cmake")
    message(">> Finder Coppied!!")
else()    
    message("\n\n!!!! Findmc.cmake Failed to Copy !!!!\n\n")
    return()
endif()

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED src/mc.cpp)
add_library(gen::mc ALIAS ${PROJECT_NAME})

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
