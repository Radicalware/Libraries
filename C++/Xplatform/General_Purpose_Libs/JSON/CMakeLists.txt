cmake_minimum_required(VERSION 3.17)

set(THIS JSON)

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
        ${installed_projects}
)

link_static(${THIS} xmap)
link_static(${THIS} xvector)
link_static(${THIS} xstring)
link_static(${THIS} Nexus)
link_static(${THIS} re2)

set(MONGO_LIBS "")
if(${debug})
    list(APPEND ${MONGO_LIBS}
        "D:/AIE/vcpkg/installed/x64-windows/debug/lib/bsoncxx.lib"
        "D:/AIE/vcpkg/installed/x64-windows/debug/lib/mongocxx.lib"
    )
else()
    list(APPEND ${MONGO_LIBS}
        "D:/AIE/vcpkg/installed/x64-windows/lib/bsoncxx.lib"
        "D:/AIE/vcpkg/installed/x64-windows/lib/mongocxx.lib"
    )
endif()

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(${THIS}
    cpprestsdk::cpprest
    cpprestsdk::cpprestsdk_zlib_internal
    cpprestsdk::cpprestsdk_brotli_internal

    nlohmann_json
    nlohmann_json::nlohmann_json

    ${MONGO_LIBS}
)

# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
INSTALL_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------
