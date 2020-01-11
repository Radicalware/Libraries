cmake_minimum_required(VERSION 3.12)

set(THIS OS)

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


add_library(${THIS} STATIC ${PROJECT_FILES})
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PUBLIC 
        ${EXT_HEADER_PATH}
        ${INC}
        ${INC}/dir_support
)


# Required Libs -------------------------------------------------------------------------

find_package(re2)
find_package(Nexus)
find_package(xvector)
find_package(xstring)

# -------------------------- INSTALL ----------------------------------------------------
include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------