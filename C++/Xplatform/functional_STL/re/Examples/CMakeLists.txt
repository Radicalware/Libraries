cmake_minimum_required(VERSION 3.12)

SET(THIS ex_re)

# -------------------------- PRE-CONFIG ---------------------------------------

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

find_package(re)
find_package(ac)

target_link_libraries( ${THIS}
    PRIVATE
        radical::re
        radical::ac
)

# -------------------------- POST-CONFIG --------------------------------------

CONFIGURE_VISUAL_STUDIO(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------