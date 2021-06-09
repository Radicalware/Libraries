cmake_minimum_required(VERSION 3.16)

set(THIS ac)

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
add_library(Radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_HEADER_PATH}
        ${INC}
)

# Required Libs -------------------------------------------------------------------------
message ("No Required Libs Needed")
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------