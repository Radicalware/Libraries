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
add_library(${LIB} STATIC ${PROJECT_FILES}) # Lib Type <<<
add_library(Radical::${LIB} ALIAS ${LIB})
target_include_directories(${LIB} PUBLIC ${installed_projects})
target_link_libraries(${LIB} ${installed_libs})
list(APPEND installed_libs "Radical::${LIB}")
add_dependencies(${THIS}   "Radical::${LIB}")

set(MongoLibs "")
if(${debug})
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/debug/lib/bsoncxx.lib")
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/debug/lib/mongocxx.lib")
else()
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/lib/bsoncxx.lib")
    target_link_libraries(${LIB} "D:/AIE/vcpkg/installed/x64-windows/lib/mongocxx.lib")
endif()

link_libraries(${LIB}
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MongoLibs}
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
