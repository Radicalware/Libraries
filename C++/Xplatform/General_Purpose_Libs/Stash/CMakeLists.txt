cmake_minimum_required(VERSION 3.17)

set(THIS ${ArgProjectName})

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

include(${VCPKG_SCRIPT})
find_program_files(project_files "${BUILD_DIR}")


add_library(${THIS} STATIC ${project_files})
add_library(Radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${VCPKG_INCLUDE}
        ${installed_projects}
)

set(MongoLibs "")
if(${debug})
    list(APPEND ${MONGO_LIBS}
        "${VCPKG_DEBUG_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_DEBUG_LIB_DIR}/mongocxx.lib"
    )
else()
    list(APPEND ${MONGO_LIBS}
        "${VCPKG_RELEASE_LIB_DIR}/bsoncxx.lib"
        "${VCPKG_RELEASE_LIB_DIR}/mongocxx.lib"
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

link_static(${THIS} Macros)
link_static(${THIS} re2)
link_static(${THIS} Nexus)
link_static(${THIS} xvector)
link_static(${THIS} xstring)
link_static(${THIS} JSON)

# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
INSTALL_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------
