cmake_minimum_required(VERSION 3.16)

set(THIS Socket)

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
    PRIVATE 
        ${EXT_HEADER_PATH}
        ${INC}
        ${INC}/Client
        ${INC}/Server
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${ST})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xvector${ST})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xstring${ST})

target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}re2${ST})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------