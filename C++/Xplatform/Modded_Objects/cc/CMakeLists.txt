cmake_minimum_required(VERSION 3.12)

set(THIS cc)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

add_library(${THIS} SHARED 

    ${INC}/${THIS}.h
    ${SRC}/${THIS}.cpp
)
add_library(radical_mod::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PUBLIC
        ${EXT_HEADER_PATH}
        ${INC}
)

# Required Libs -------------------------------------------------------------------------
message ("No Required Libs Needed")
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------
