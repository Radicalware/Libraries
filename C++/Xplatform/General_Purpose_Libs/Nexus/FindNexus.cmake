cmake_minimum_required(VERSION 3.10)

set(LIB Nexus)

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    if("${BUILD_TYPE}" STREQUAL "Release")
        add_definitions( "-O2" )
    endif()
else()
    set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -fPIC -pthread")
    if("${BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(Nexus_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} 
    STATIC 
        ${Nexus_DIR}/src/CPU_Threads.cpp
        ${Nexus_DIR}/include/CPU_Threads.h

        ${Nexus_DIR}/src/Task.cpp
        ${Nexus_DIR}/include/Task.h

        ${Nexus_DIR}/src/Job.cpp
        ${Nexus_DIR}/include/Job.h

        ${Nexus_DIR}/src/${LIB}.cpp
        ${Nexus_DIR}/include/${LIB}.h

        ${Nexus_DIR}/src/${LIB}_void.cpp
        ${Nexus_DIR}/include/${LIB}_void.h

        ${Nexus_DIR}/src/${LIB}_T.cpp
        ${Nexus_DIR}/include/${LIB}_T.h
)
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE        
        ${Nexus_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
