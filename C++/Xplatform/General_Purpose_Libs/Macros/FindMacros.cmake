cmake_minimum_required(VERSION 3.17)

set(LIB Macros)

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

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
install_static_lib(${LIB})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
