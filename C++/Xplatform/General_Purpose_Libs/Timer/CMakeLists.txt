cmake_minimum_required(VERSION 3.16)

set(THIS Timer)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Project")

add_library(${THIS} SHARED ${PROJECT_FILES})
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PUBLIC
        ${INC}
        ${EXT_HEADER_PATH}
)


# Required Libs -------------------------------------------------------------------------

find_package(Nexus)
find_package(xvector)
find_package(xstring)
find_package(xmap)
find_package(re2)

target_link_libraries(${THIS} 
    Radical::Nexus
    Radical::xvector
    Radical::xstring
    Radical::xmap

    Radical_Mod::re2
)

# -------------------------- INSTALL ----------------------------------------------------
include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------