cmake_minimum_required(VERSION 3.16)

SET(THIS ex_ac)

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

find_package(re)
find_package(ac)

target_link_libraries(${THIS} 
    Radical::ac
    Radical::re
)
# -------------------------- POST-CONFIG --------------------------------------

CONFIGURE_VISUAL_STUDIO_SOLUTION(${SOLUTION_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------
