
# MakeFile for ac.h >> acgex

cmake_minimum_required(VERSION 3.10)

project (ac)


execute_process (COMMAND bash -c "rm -f /usr/share/cmake/Modules/Findac.cmake")
execute_process (
    COMMAND bash -c "cp ../Findac.cmake /usr/share/cmake/Modules/"
    OUTPUT_VARIABLE outVar
)

if(EXISTS "/usr/share/cmake/Modules/Findac.cmake")
    message(">> Finder Coppied!!")
else()    
    message("\n\n!!!! Findac.cmake Failed to Copy !!!!\n\n")
    return()
endif()

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED src/ac.cpp)
add_library(gen::ac ALIAS ${PROJECT_NAME})

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
    DESTINATION     code/ac
)
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/ac
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           Findac.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
