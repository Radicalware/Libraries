cmake_minimum_required(VERSION 3.12)

SET(THIS ex_OS)

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

    ${INC}/ex_bash_style.h
    ${INC}/ex_file_managment.h
    ${INC}/ex_open_n_delete.h
)

find_package(Nexus)
find_package(xvector)
find_package(xstring)
find_package(xmap)
find_package(OS)

target_link_libraries( ${THIS}
    PRIVATE        
        radical::Nexus
        radical::xvector
        radical::xstring
        radical::xmap
        radical::OS
)

# -------------------------- POST-CONFIG --------------------------------------

CONFIGURE_VISUAL_STUDIO(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------
