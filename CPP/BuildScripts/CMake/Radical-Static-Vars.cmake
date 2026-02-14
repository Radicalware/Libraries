
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
    set(Release ON)
    set(debug OFF)
    set(Debug OFF)
    add_compile_definitions("ReleaseOn")
else()
    set(release OFF)
    set(Release OFF)
    set(debug ON)
    set(Debug ON)
    add_compile_definitions("DebugOn")
endif()

set(CMAKE_CXX_IGNORE_EXTENSIONS      "${CMAKE_CXX_IGNORE_EXTENSIONS};txt;rc")
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS "${CMAKE_CXX_SOURCE_FILE_EXTENSIONS};cuh;cu")

set(CUDA_VERSION "12.4")
set(CUDA_GPU 61)

cmake_policy(SET CMP0167 NEW)

if(WIN32) # ----------------------------------------------------------------------------

    set(OS_TYPE "Windows")
    set(IsWindows ON)
    
    # Dynamically find the most recent MSVC version
    file(GLOB MSVC_VERSIONS "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/*/")
    if(MSVC_VERSIONS)
        list(SORT MSVC_VERSIONS)
        list(POP_BACK MSVC_VERSIONS MSVC_LATEST)
        set(ATL_INCLUDE "${MSVC_LATEST}/atlmfc/include")
        message(STATUS "Found MSVC ATL path: ${ATL_INCLUDE}")
    else()
        message(FATAL_ERROR "MSVC installation not found at expected location")
    endif()
    include_directories(${ATL_INCLUDE})
    # set(MSVC_TOOLSET_VERSION "142")
    # set(WINDOWS_SDK "10.0.18362.0")
    # set(CMAKE_SYSTEM_VERSION ${WINDOWS_SDK})
    # set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION ${WINDOWS_SDK})

    add_definitions(-DUsingMSVC)

    # RUN: vcpkg integrate install

    # Get the VCPKG_ROOT environment variable
    if(DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{VCPKG_ROOT})
        string(REPLACE "\\" "/" VCPKG_ROOT ${VCPKG_ROOT})
        message(STATUS "VCPKG_ROOT is set to ${VCPKG_ROOT}")
    else()
        message(FATAL_ERROR "VCPKG_ROOT environment variable is not set")
    endif()
    
    set(Qt6_Base_DIR    "C:/Qt/6.8.2/msvc2022_64/lib/cmake/Qt6")
    set(Qt6_DIR         "${Qt6_Base_DIR}/Qt6")
    set(CMAKE_PATH      "C:/Program Files/CMake/share/cmake-$ENV{CMAKE_VERSION}/Modules")
    set(RADICAL_BASE    "C:/Source/CMake/Radicalware")

    list(APPEND "D:/libs/Felgo/Felgo/mingw_64/lib/cmake/Felgo")
    list(APPEND "D:/libs/Felgo/Felgo/mingw_64/lib/cmake/FelgoHotReload")

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

    set(CMAKE_CXX_STANDARD 20) 
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    set(C_ARGS   "${CPP_ARGS} ${C_ARGS}")
    set(C_ARGS "  ${C_ARGS}   /std:c17")
    set(CPP_ARGS "${CPP_ARGS} /std:c++latest /EHsc /bigobj /MP /DPARALLEL")
    # MP = Multp Processing = build using mutiple threads
    # TP = Treat all files as cPP files
    # /DPARALLEL supports C++latest threading
    # add_link_options("/ignore:4099") # Ignore PDB Warnings
    # add_link_options("/ignore:4204") # Ignore PDB Warnings
    # add_link_options("/INCREMENTAL:NO")

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${Qt6_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${RADICAL_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

    PrintList(CMAKE_MODULE_PATH)
    
    list(APPEND CMAKE_PREFIX_PATH "${Qt6_Base_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${Qt6_DIR}")

    set(PF  "")    # Prefix
    set(ST  "lib") # STatic
    set(SH  "dll") # SHared
    set(OBJ "obj") # OBJect 

    set(VCPKG_TARGET_TRIPLET  "x64-windows")

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
    set(CPP_ARGS "${CPP_ARGS} -std=c++latest")

    set(CMAKE_PATH                "/usr/share/cmake-$ENV{CMAKE_VERSION}/Modules")
    list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-$ENV{CMAKE_VERSION}/Utilities/cmlibarchive/build/cmake")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_PATH})
    list(APPEND CMAKE_MODULE_PATH ${RADICAL_PATH})

    set(PF  "lib")
    set(ST  "a")
    set(SH  "so")
    set(OBJ "cpp.o")
endif() # ----------------------------------------------------------------------------

# if(NOT DEFINED CUDA_VERSION)
    set(VCPKG_RELEASE_LIB_DIR "${VCPKG_ROOT}/installed/x64-windows/lib")
    set(VCPKG_DEBUG_LIB_DIR   "${VCPKG_ROOT}/installed/x64-windows/lib")
    set(VCPKG_SCRIPT          "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE  "${VCPKG_SCRIPT}")
# endif()
if(WIN32)
    set(VCPKG_INCLUDE         "${VCPKG_ROOT}/installed/x64-windows/include")
else()
    set(VCPKG_INCLUDE         "${VCPKG_ROOT}/installed/x64-linux/include")
endif()

if(debug)
    message("Link Debug Iterator 2 (aka ON)")
    #add_definitions(-D_ITERATOR_DEBUG_LEVEL=2)
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=2)
    #add_compile_definitions(-D_ITERATOR_DEBUG_LEVEL=2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_ITERATOR_DEBUG_LEVEL=2")
else()
    message("Link Debug Iterator 0 (aka OFF)")
    #add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    #add_compile_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_ITERATOR_DEBUG_LEVEL=0")
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
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}")
    else()
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}")
    endif()
else()
    set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}")    
endif()

MakeDir("${OUTPUT_DIR}")
MakeDir("${OUTPUT_DIR}/lib")
MakeDir("${OUTPUT_DIR}/bin")

if(NOT ${BuildAll})
    MakeDir("${EXT_BIN_PATH}/lib")
endif()

message(" running toolchain >> ${CMAKE_TOOLCHAIN_FILE}")
include("${CMAKE_TOOLCHAIN_FILE}")

# --------------- DON'T MODIFY (CALCULATED) ------------------------------------------

