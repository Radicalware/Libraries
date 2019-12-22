cmake_minimum_required(VERSION 3.12)

SET(THIS ex_Nexus)

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
)

find_package(Nexus)
find_package(xvector)
find_package(xstring)
find_package(xmap)
find_package(Timer)

target_link_libraries( ${THIS}
    PRIVATE
        radical::Nexus
        radical::xvector
        radical::xstring
        radical::xmap
        radical::Timer
)

# -------------------------- END ----------------------------------------------
