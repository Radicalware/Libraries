cmake_minimum_required(VERSION 3.17)

SET(THIS ex_CudaBridge)

# -------------------------- PRE-CONFIG ---------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")

# -------------------------- BUILD --------------------------------------------

project(${THIS} LANGUAGES CXX CUDA)

find_package(PrepareCuda113)

find_program_files(SOLUTION_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Solution")
add_executable(${THIS} ${SOLUTION_FILES})

find_package(CudaBridge)

target_include_directories(${THIS} PRIVATE
    ${installed_projects}
)

target_link_libraries(${THIS}

    Radical::CudaBridge
)

set_target_properties(${THIS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -------------------------- POST-CONFIG --------------------------------------

INSTALL_VISUAL_STUDIO_SOLUTION(${SOLUTION_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END ----------------------------------------------