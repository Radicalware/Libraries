cmake_minimum_required(VERSION 3.17)

set(LIB OS)

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
add_library(${LIB} MODULE ${PROJECT_FILES})
set_target_properties(${LIB} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)
add_library(Radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB} PUBLIC
    ${installed_projects}
)

link_static(${LIB} re2)
link_static(${LIB} Nexus)
link_static(${LIB} xvector)
link_static(${LIB} xstring)
link_static(${LIB} xmap)

link_dynamic(${THIS} ${LIB})

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
