# Copyright 2015 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Old enough to support Ubuntu Xenial.
cmake_minimum_required(VERSION 3.5.1)


set(LIB re2)
list(APPEND SHARED_LIB_LST ${LIB})

# -------------------------- CONFIGURATION ------------------------------------
set(RE2_DIR ${PROJECT_DIR}/${LIB})
set(INC     ${RE2_DIR}/include)
set(SRC     ${RE2_DIR}/src)
# -------------------------- BUILD --------------------------------------------

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

include(GNUInstallDirs)

#option(BUILD_SHARED_LIBS "build shared libraries" OFF)
option(USEPCRE "use PCRE in tests and benchmarks" OFF)

# CMake seems to have no way to enable/disable testing per subproject,
# so we provide an option similar to BUILD_TESTING, but just for RE2.
option(RE2_BUILD_TESTING "enable testing for RE2" OFF)

set(EXTRA_TARGET_LINK_LIBRARIES)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(MSVC_VERSION LESS 1900)
        message(FATAL_ERROR "you need Visual Studio 2015 or later")
    endif()
    if(BUILD_SHARED_LIBS)
        # See http://www.kitware.com/blog/home/post/939 for details.
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    # CMake defaults to /W3, but some users like /W4 (or /Wall) and /WX,
    # so we disable various warnings that aren't particularly helpful.
    add_compile_options(/wd4100 /wd4201 /wd4456 /wd4457 /wd4702 /wd4815)
    # Without a byte order mark (BOM), Visual Studio assumes that the source
    # file is encoded using the current user code page, so we specify UTF-8.
    add_compile_options(/utf-8)
endif()

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE -DSTRICT -DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
elseif(UNIX)
    list(APPEND EXTRA_TARGET_LINK_LIBRARIES -pthread)
endif()

if(USEPCRE)
    add_definitions(-DUSEPCRE)
    list(APPEND EXTRA_TARGET_LINK_LIBRARIES pcre)
endif()

SUBDIRLIST(PROJECT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Project")

add_library(re2 ${RE2_HEADERS} ${RE2_SOURCES})
add_library(radical_mod::re2 ALIAS re2)

include_directories(
    ${RE2_DIR}/include
)

# -------------------------- BUILD --------------------------------------------
list(APPEND PROJECT_FILES ${RE2_HEADERS} ${RE2_SOURCES})
CONFIGURE_VISUAL_STUDIO(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
