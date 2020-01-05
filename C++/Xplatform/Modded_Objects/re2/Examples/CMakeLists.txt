cmake_minimum_required(VERSION 3.12)

SET(THIS ex_re2)

# -------------------------- PRE-CONFIG ---------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")

# -------------------------- BUILD --------------------------------------------

project (${THIS})

SUBDIRLIST(PROJECT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Solution")

add_executable(${THIS}
    ${PROJECT_FILES}
)

find_package(re2)
find_package(Nexus)
find_package(xvector)
find_package(xstring)

target_link_libraries( ${THIS} PRIVATE
    radical::Nexus
    radical::xvector
    radical::xstring
    radical_mod::re2
)

# -------------------------- POST-CONFIG --------------------------------------

CONFIGURE_VISUAL_STUDIO(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------
