cmake_minimum_required(VERSION 3.17)

set(LIB CC)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND SHARED_LIB_LST ${LIB})

if(${release} AND NOT ${build_all})
    link_dynamic(${THIS} ${LIB})
    return()
endif()
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
find_program_files(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} MODULE ${PROJECT_FILES})
add_library(Radical_Mod::${LIB} ALIAS ${LIB})

include_directories(${LIB} PUBLIC
    ${installed_projects}
)

link_dynamic(${THIS} ${LIB})
set_target_properties(${LIB} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
install_dynamic_lib(${LIB})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------