cmake_minimum_required(VERSION 3.17)

set(LIB Stash)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND STATIC_LIB_LST ${LIB})

list(APPEND installed_projects "${PROJECT_DIR}/${LIB}/include")

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

link_static(${LIB} xmap)
link_static(${LIB} xstring)
link_static(${LIB} xvector)
link_static(${LIB} Nexus)
link_static(${LIB} re2)
link_static(${LIB} JSON)

set(MongoLibs "")
if(${debug})
    list(APPEND ${MongoLibs}
        "D:/AIE/vcpkg/installed/x64-windows/debug/lib/bsoncxx.lib"
        "D:/AIE/vcpkg/installed/x64-windows/debug/lib/mongocxx.lib"
    )
else()
    list(APPEND ${MongoLibs}
        "D:/AIE/vcpkg/installed/x64-windows/lib/bsoncxx.lib"
        "D:/AIE/vcpkg/installed/x64-windows/lib/mongocxx.lib"
    )
endif()

link_libraries(
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MongoLibs}
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
install_static_lib(${LIB})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
