cmake_minimum_required(VERSION 3.17)

set(THIS xstring)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

find_program_files(project_files "${BUILD_DIR}")

add_library(${THIS} STATIC ${project_files})
add_library(Radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${installed_projects}
)

link_static(${THIS} xvector)
link_static(${THIS} Nexus)
link_static(${THIS} re2)

# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
INSTALL_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------