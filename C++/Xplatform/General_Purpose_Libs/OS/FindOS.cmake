cmake_minimum_required(VERSION 3.17)

set(LIB OS)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND SHARED_LIB_LST ${LIB})

if(${release} AND NOT ${build_all})
    link_dynamic(${THIS} ${LIB})
    return()
endif()
# -------------------------- BUILD --------------------------------------------

UNSET(project_files)
find_program_files(project_files "${PROJECT_DIR}/${LIB}")

add_library(${LIB} MODULE ${project_files})
add_library(Radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PUBLIC
    ${installed_projects}
)

link_static(${LIB} xmap)
link_static(${LIB} xstring)
link_static(${LIB} xvector)
link_static(${LIB} Nexus)
link_static(${LIB} re2)

link_dynamic(${THIS} ${LIB})
set_target_properties(${LIB} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${project_files})
install_dynamic_lib(${LIB})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
