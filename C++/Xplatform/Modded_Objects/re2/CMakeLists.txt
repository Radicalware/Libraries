cmake_minimum_required(VERSION 3.16)

set(THIS re2)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project(${THIS} CXX)

# Copyright 2015 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Old enough to support Ubuntu Xenial.

set(RE2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Project")


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

UNSET(PROJECT_FILES)
set(PROJECT_FILES
    ${RE2_DIR}/include/re2/bitmap256.h
    ${RE2_DIR}/include/re2/filtered_re2.h
    ${RE2_DIR}/include/re2/pod_array.h
    ${RE2_DIR}/include/re2/prefilter.h
    ${RE2_DIR}/include/re2/prefilter_tree.h
    ${RE2_DIR}/include/re2/prog.h
    ${RE2_DIR}/include/re2/re2.h
    ${RE2_DIR}/include/re2/regexp.h
    ${RE2_DIR}/include/re2/set.h
    ${RE2_DIR}/include/re2/sparse_array.h
    ${RE2_DIR}/include/re2/sparse_set.h
    ${RE2_DIR}/include/re2/stringpiece.h
    ${RE2_DIR}/include/re2/unicode_casefold.h
    ${RE2_DIR}/include/re2/unicode_groups.h
    ${RE2_DIR}/include/re2/walker-inl.h
    ${RE2_DIR}/include/util/benchmark.h
    ${RE2_DIR}/include/util/flags.h
    ${RE2_DIR}/include/util/logging.h
    ${RE2_DIR}/include/util/malloc_counter.h
    ${RE2_DIR}/include/util/mix.h
    ${RE2_DIR}/include/util/mutex.h
    ${RE2_DIR}/include/util/pcre.h
    ${RE2_DIR}/include/util/strutil.h
    ${RE2_DIR}/include/util/test.h
    ${RE2_DIR}/include/util/utf.h
    ${RE2_DIR}/include/util/util.h
    ${RE2_DIR}/src/re2/bitstate.cpp
    ${RE2_DIR}/src/re2/compile.cpp
    ${RE2_DIR}/src/re2/dfa.cpp
    ${RE2_DIR}/src/re2/filtered_re2.cpp
    ${RE2_DIR}/src/re2/mimics_pcre.cpp
    ${RE2_DIR}/src/re2/nfa.cpp
    ${RE2_DIR}/src/re2/onepass.cpp
    ${RE2_DIR}/src/re2/parse.cpp
    ${RE2_DIR}/src/re2/perl_groups.cpp
    ${RE2_DIR}/src/re2/prefilter.cpp
    ${RE2_DIR}/src/re2/prefilter_tree.cpp
    ${RE2_DIR}/src/re2/prog.cpp
    ${RE2_DIR}/src/re2/re2.cpp
    ${RE2_DIR}/src/re2/regexp.cpp
    ${RE2_DIR}/src/re2/set.cpp
    ${RE2_DIR}/src/re2/simplify.cpp
    ${RE2_DIR}/src/re2/stringpiece.cpp
    ${RE2_DIR}/src/re2/tostring.cpp
    ${RE2_DIR}/src/re2/unicode_casefold.cpp
    ${RE2_DIR}/src/re2/unicode_groups.cpp
    ${RE2_DIR}/src/util/pcre.cpp
    ${RE2_DIR}/src/util/rune.cpp
    ${RE2_DIR}/src/util/strutil.cpp
)

add_library(${THIS} STATIC ${PROJECT_FILES})
add_library(Radical_Mod::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${installed_projects}
)

if(WIN32)
    # Windows is auto-installed later; no mod installs needed
else()
    set(RE2_OBJ_FILE_PATH "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/CMakeFiles/${THIS}.dir/Project/src/re2/${THIS}.${OBJ}")
    
    install(
        FILES ${RE2_OBJ_FILE_PATH}
        CONFIGURATIONS  ${BUILD_TYPE}
        DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
        OPTIONAL
    )
endif()


# Required Libs -------------------------------------------------------------------------
# No Required Libs Needed
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
INSTALL_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(REMOVE "${INSTALL_PREFIX}/Libraries/Projects/re2/include/re2.h")
file(REMOVE "${INSTALL_PREFIX}/Libraries/Projects/re2/src/re2.cpp")

# -------------------------- END --------------------------------------------------------
