﻿cmake_minimum_required(VERSION 3.17)

set(LIB SYS)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND SHARED_LIB_LST ${LIB})

list(APPEND installed_projects   "${PROJECT_DIR}/${LIB}/include")

if(${release} AND NOT ${build_all})
    link_dynamic(${THIS} ${LIB})
    return()
endif()
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
find_program_files(PROJECT_FILES "${PROJECT_DIR}/${LIB}")
add_library(${LIB} MODULE ${PROJECT_FILES}) # Lib Type <<<
add_library(Radical::${LIB} ALIAS ${LIB})
target_include_directories(${LIB} PUBLIC ${installed_projects})
target_link_libraries(${LIB} ${installed_libs})
add_dependencies(${THIS}   "Radical::${LIB}")

link_dynamic(${THIS} ${LIB})
set_target_properties(${LIB} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
