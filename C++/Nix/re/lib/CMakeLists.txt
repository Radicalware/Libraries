# MakeFile for re.h

cmake_minimum_required(VERSION 3.10)

project (re)


execute_process (COMMAND bash -c "rm -f /usr/share/cmake/Modules/Findre.cmake")
execute_process (
    COMMAND bash -c "cp ../Findre.cmake /usr/share/cmake/Modules/"
    OUTPUT_VARIABLE outVar
)

if(EXISTS "/usr/share/cmake/Modules/Findre.cmake")
    message(">> Finder Coppied!!")
else()    
    message("\n\n!!!! Findre.cmake Failed to Copy !!!!\n\n")
    return()
endif()

add_definitions(
    -Wfatal-errors
    -std=c++17
    -O2
)

add_library(${PROJECT_NAME} SHARED src/re.cpp)
add_library(gen::re ALIAS ${PROJECT_NAME})

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

# re files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include 
    DESTINATION     code/re
)
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/include
    DESTINATION     .
)

# Source files
install (   
    DIRECTORY       ${PROJECT_SOURCE_DIR}/src 
    DESTINATION     code/re
)

# Config
set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
install (   
    FILES           Findre.cmake 
    DESTINATION     ${CMAKE_FIND_PATH}
)
