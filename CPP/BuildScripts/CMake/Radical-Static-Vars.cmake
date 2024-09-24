
if(UNIX)
    message("These libraries are currently under rapid development so will only officially support Windows")
    message("Official Unix support will return later, remove this hault to use on Linux")
    message("It's located in Radical-Static-Vars.cmake at the first conditional Statment")
    message("Thanks for your patience!")
    exit()
endif()

# ------------------------- MODIFY VALUES BELOW ----------------------------------------

set(CPP_ARGS "")
set(C_ARGS   "")

if(BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
else()
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(release ON)
    set(debug OFF)
    add_compile_definitions("ReleaseOn")
else()
    set(release OFF)
    set(debug ON)
    add_compile_definitions("DebugOn")
endif()

set(CMAKE_CXX_IGNORE_EXTENSIONS      "${CMAKE_CXX_IGNORE_EXTENSIONS};txt;rc")
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS "${CMAKE_CXX_SOURCE_FILE_EXTENSIONS};cuh;cu")

if(WIN32) # ----------------------------------------------------------------------------

    set(OS_TYPE "Windows")
    set(IsWindows ON)
    

    # set(MSVC_TOOLSET_VERSION "142")
    # set(WINDOWS_SDK "10.0.18362.0")
    # set(CMAKE_SYSTEM_VERSION ${WINDOWS_SDK})
    # set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION ${WINDOWS_SDK})

    add_definitions(-DUsingMSVC)

    # RUN: vcpkg integrate install
    set(VCPKG_ROOT            "C:/Source/Git/vcpkg")
    set(CMAKE_TOOLCHAIN_FILE    ${VCPKG_SCRIPT})
    set(Qt6_DIR         "C:/Qt/6.6.1/msvc2019_64/lib/cmake/Qt6")
    set(CMAKE_PATH      "C:/Program Files/CMake/share/cmake-$ENV{CMAKE_VERSION}/Modules")
    set(RADICAL_BASE    "C:/Source/CMake/Radicalware")

    SET(INSTALL_PREFIX "${RADICAL_BASE}")
    SET(INSTALL_DIR    "${RADICAL_BASE}/Libraries/Projects")

    
    link_directories("$ENV{VULKAN_SDK}/lib")
    set(VULKAN_CMAKE_DIR   "C:/a/Vulkan-Hpp")
    set(VULKAN_INCLUDE_DIR "${VULKAN_CMAKE_DIR}/Vulkan-Headers/include")
    set(VULKAN_HPP_PATH    "${VULKAN_INCLUDE_DIR}")
    find_path(VULKAN_HEADERS_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR}/vulkan.hpp")

    # BUILD_DIR   = ready to install cpp files
    # INSTALL_DIR = install location of those cpp files
    FindProgramFiles(RADICAL_PROGRAM_FILES "${INSTALL_DIR}")
    set(RE2_DIR "C:/Source/Git/re2")

    set(C_ARGS   "${CPP_ARGS} ${C_ARGS}")
    set(C_ARGS "  ${C_ARGS}   /std:c17")
    set(CMAKE_CXX_STANDARD 20)
    set(CPP_ARGS "${CPP_ARGS} /std:c++20 /EHsc /MP /DPARALLEL")
    # MP = Multp Processing = build using mutiple threads
    # TP = Treat all files as cPP files
    # /DPARALLEL supports C++20 threading
    # add_link_options("/ignore:4099") # Ignore PDB Warnings
    # add_link_options("/ignore:4204") # Ignore PDB Warnings
    # add_link_options("/INCREMENTAL:NO")

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${Qt6_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${RADICAL_PATH}")

    PrintList(CMAKE_MODULE_PATH)

    set(PF  "")    # Prefix
    set(ST  "lib") # STatic
    set(SH  "dll") # SHared
    set(OBJ "obj") # OBJect 


else() # -----------------------------------------------------------------------------
    SET(OS_TYPE "Nix")
    set(IsNix ON)

    SET(RADICAL_BASE   "/opt/Radicalware")
    SET(INSTALL_PREFIX "${RADICAL_BASE}")
    SET(INSTALL_DIR "   ${RADICAL_BASE}/Libraries/Projects")
    FindProgramFiles(RADICAL_PROGRAM_FILES "${INSTALL_DIR}")

    set(CPP_ARGS " -Wfatal-errors -finput-charset=UTF-8 -fPIC -pthread")
    set(CPP_ARGS "${CPP_ARGS} -Wno-unused-result")
    set(C_ARGS   "${CPP_ARGS} ${C_ARGS}")
    set(C_ARGS   "${C_ARGS}   -std:c17")
    set(CPP_ARGS "${CPP_ARGS} -std=c++20")

    set(CMAKE_PATH                "/usr/share/cmake-$ENV{CMAKE_VERSION}/Modules")
    list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-$ENV{CMAKE_VERSION}/Utilities/cmlibarchive/build/cmake")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_PATH})
    list(APPEND CMAKE_MODULE_PATH ${RADICAL_PATH})

    set(PF  "lib")
    set(ST  "a")
    set(SH  "so")
    set(OBJ "cpp.o")
endif() # ----------------------------------------------------------------------------

if(NOT DEFINED CUDA_VERSION)
    set(VCPKG_RELEASE_LIB_DIR "${VCPKG_ROOT}/installed/x64-windows/lib")
    set(VCPKG_DEBUG_LIB_DIR   "${VCPKG_ROOT}/installed/x64-windows/lib")
    set(VCPKG_SCRIPT          "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(VCPKG_INCLUDE         "${VCPKG_ROOT}/installed/x64-windows/include")
endif()

if(${debug})
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=2)
else()
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
endif()

# ------------------------- MODIFY VALUES ABOVE --------------------------------------

# --------------- DON'T MODIFY (CALCULATED) ------------------------------------------

if(BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
else()
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(WIN32)
        set(${CPP_ARGS} " ${CPP_ARGS} /O2")
        set(C_ARGS      " ${C_ARGS}   /O2")
    else()
        set(${CPP_ARGS} " ${CPP_ARGS} -O2")
        set(C_ARGS      " ${C_ARGS}   -O2")
    endif()
elseif(else) # debug
    if(WIN32)
        set(${CPP_ARGS} " ${CPP_ARGS} /Od /Zi /MDd")
        set(C_ARGS      " ${C_ARGS}   /Od /Zi /MDd")
    else()
        set(${CPP_ARGS} " ${CPP_ARGS} -g3 -ggdb")
        set(C_ARGS      " ${C_ARGS}   -g3 -ggdb")
    endif()
endif()

if(BuildAll STREQUAL "ON" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BuildAll ON)
else()
    set(BuildAll OFF)
endif()

set(CMAKE_CXX_FLAGS ${CPP_ARGS})
#set(CMAKE_C_FLAGS   ${C_ARGS})

set(EXT_HEADER_PATH ${RADICAL_BASE}/Libraries/Headers)
set(EXT_BIN_PATH    ${RADICAL_BASE}/Libraries/Build/${BUILD_TYPE})

set(PROJECT_DIR ${INSTALL_PREFIX}/Libraries/Projects)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(installed_libs) # Used to link targets and set dependencies

include_directories("${VCPKG_INCLUDE}")
include_directories("${VULKAN_INCLUDE_DIR}")

if(WIN32)
    if(${IsApp})
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/lib")
    else()
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}")
    endif()
else()
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/lib")    
endif()

if(${BuildAll})
    MakeDir("${OUTPUT_DIR}")
else()
    MakeDir("${EXT_BIN_PATH}/lib")
endif()

# --------------- DON'T MODIFY (CALCULATED) ------------------------------------------

