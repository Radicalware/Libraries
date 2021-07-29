cmake_minimum_required(VERSION 3.17)

set(LIB JSON)

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

target_include_directories(${LIB} PUBLIC
    ${installed_projects}
)

link_static(${LIB} xmap)
link_static(${LIB} xstring)
link_static(${LIB} xvector)
link_static(${LIB} Nexus)
link_static(${LIB} re2)

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(${LIB}
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
install_static_lib(${LIB})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
