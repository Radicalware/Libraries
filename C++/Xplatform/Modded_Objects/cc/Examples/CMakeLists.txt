cmake_minimum_required(VERSION 3.12)

SET(THIS ex_cc)

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

find_package(cc)

target_link_libraries( ${THIS}
    PRIVATE
        radical_mod::cc
)

# -------------------------- END ----------------------------------------------
