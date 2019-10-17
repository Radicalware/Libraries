cmake_minimum_required(VERSION 3.10)

set(LIB Threader)

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling ${THIS} with -O2 ${BUILD_TYPE}")
        add_definitions( "-O2" )
    endif()
else()
    set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -fPIC -pthread")
    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling with -O2 ${BUILD_TYPE}")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(Threader_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} 
    STATIC 
        ${Threader_DIR}/src/CPU_Threads.cpp
        ${Threader_DIR}/include/CPU_Threads.h

        ${Threader_DIR}/src/Task.cpp
        ${Threader_DIR}/include/Task.h

        ${Threader_DIR}/src/Job.cpp
        ${Threader_DIR}/include/Job.h

        ${Threader_DIR}/src/${LIB}.cpp
        ${Threader_DIR}/include/${LIB}.h

        ${Threader_DIR}/src/${LIB}_void.cpp
        ${Threader_DIR}/include/${LIB}_void.h

        ${Threader_DIR}/src/${LIB}_T.cpp
        ${Threader_DIR}/include/${LIB}_T.h
)
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE        
        ${Threader_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
