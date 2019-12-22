cmake_minimum_required(VERSION 3.12)

set(THIS Nexus)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

add_library(${THIS} STATIC

    ${SRC}/NX_Threads.cpp
    ${INC}/NX_Threads.h

    ${SRC}/NX_Mutex.cpp
    ${INC}/NX_Mutex.h

    ${SRC}/Task.cpp
    ${INC}/Task.h

    ${SRC}/Job.cpp
    ${INC}/Job.h

    ${SRC}/${THIS}.cpp
    ${INC}/${THIS}.h

    ${SRC}/${THIS}_void.cpp
    ${INC}/${THIS}_void.h

    ${SRC}/${THIS}_T.cpp
    ${INC}/${THIS}_T.h
)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_HEADER_PATH}
        ${INC}
)

# Required Libs -------------------------------------------------------------------------
message ("No Required Libs Needed")
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------
