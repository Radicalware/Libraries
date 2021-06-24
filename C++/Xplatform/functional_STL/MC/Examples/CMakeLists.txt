cmake_minimum_required(VERSION 3.17)

SET(THIS ex_MC)

# -------------------------- PRE-CONFIG ---------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")

# -------------------------- BUILD --------------------------------------------

project (${THIS})

find_program_files(SOLUTION_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Solution")
add_executable(${THIS} ${SOLUTION_FILES})

find_package(AC)
find_package(MC)

target_include_directories(${THIS} PRIVATE
        ${installed_projects}
)

if(${debug} OR ${build_all})
    target_link_libraries(${THIS}

        Radical::AC
        Radical::MC
    )
endif()

# -------------------------- POST-CONFIG --------------------------------------

INSTALL_VISUAL_STUDIO_SOLUTION(${SOLUTION_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------
