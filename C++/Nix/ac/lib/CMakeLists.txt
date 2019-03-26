
# CMakeFile for ac.h

cmake_minimum_required(VERSION 3.10)

project (ac)

if(EXISTS "${PROJECT_SOURCE_DIR}/Findac.cmake")
    message(">> Finder Ready for Copy!!")
else()
    message("\n\n!!!! Findac.cmake NOT FOUND !!!!\n\n")
    return()
endif()

set(CMAKE_FIND_PATH "/usr/share/cmake/Modules")
file(REMOVE ${CMAKE_FIND_PATH}/Findac.cmake)

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

# ac files
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
