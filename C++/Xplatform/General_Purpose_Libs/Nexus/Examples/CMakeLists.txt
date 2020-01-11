cmake_minimum_required(VERSION 3.12)

SET(THIS ex_Nexus)

# -------------------------- PRE-CONFIG ---------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")

# -------------------------- BUILD --------------------------------------------

project (${THIS})

SUBDIRLIST(SOLUTION_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Solution")

add_executable(${THIS} ${SOLUTION_FILES})

find_package(Nexus)
find_package(re2)
find_package(xvector)
find_package(xstring)
find_package(xmap)
find_package(Timer)

# -------------------------- POST-CONFIG --------------------------------------

CONFIGURE_VISUAL_STUDIO_SOLUTION(${SOLUTION_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------