cmake_minimum_required(VERSION 3.12)

set(THIS ex_xstring)

# -------------------------- CONFIGURATION ------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")

# -------------------------- BUILD --------------------------------------------

project (${THIS})

add_executable(${THIS}
    ${SRC}/${THIS}.cpp

    ${INC}/Full.h
    ${SRC}/Full.cpp

    ${INC}/Option.h
    ${SRC}/Option.cpp
)

find_package(Nexus)
find_package(xvector)
find_package(xstring)

target_link_libraries( ${THIS}
    PRIVATE
        radical::Nexus
        radical::xvector
        radical::xstring
)

# -------------------------- END ----------------------------------------------
