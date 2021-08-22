cmake_minimum_required(VERSION 3.17)

set(LIB xmap)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND STATIC_LIB_LST ${LIB})

list(APPEND installed_projects   "${PROJECT_DIR}/${LIB}/include")

if(${release} AND NOT ${build_all})
    link_static(${THIS} ${LIB})
    return()
endif()
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
find_program_files(PROJECT_FILES "${PROJECT_DIR}/${LIB}")
add_library(${LIB} STATIC ${PROJECT_FILES})
add_library(Radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB} PRIVATE
    ${installed_projects}
)

link_static(${LIB} xstring)
link_static(${LIB} xvector)
link_static(${LIB} Nexus)
link_static(${LIB} re2)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
