# Copyright 2022 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was re2Config.cmake.in                           #######

include("${RADICAL_PATH}/Radical-Static-Libs-Methods.cmake")

# if(${debug})
#     add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
# endif()

if(WIN32) 
    set(RE2_LIB "re2")

    list(APPEND InstalledIncludeDirs "${RE2_DIR}")
    if(${debug})
        set(RE2_LIB_PATH "${RE2_DIR}/buildit/Debug/${PF}${RE2_LIB}.${ST}")
        link_directories("${RE2_DIR}/buildit/Debug")
    else()
        set(RE2_LIB_PATH "${RE2_DIR}/buildit/Release/${PF}${RE2_LIB}.${ST}")
        link_directories("${RE2_DIR}/buildit/Release")
    endif()
    message("re2::re2 >> ${RE2_LIB_PATH}")

    add_library(re2 STATIC IMPORTED)
    set_target_properties(re2 PROPERTIES IMPORTED_LOCATION "${RE2_LIB_PATH}")
    add_library(re2::re2 ALIAS re2)

    set_target_properties(re2 PROPERTIES LINKER_LANGUAGE CXX)

    message("Copying RE2 --> ${OUTPUT_DIR}")
    if(${BuildAll})
        MakeBinaryFileCopy(${RE2_LIB_PATH} "${OUTPUT_DIR}/lib/${PF}${RE2_LIB}.${ST}")
    else()
        MakeBinaryFileCopy(${RE2_LIB_PATH} "${EXT_BIN_PATH}/lib/${PF}${RE2_LIB}.${ST}")
    endif()
else()
    message("No Linux Support")
endif()
