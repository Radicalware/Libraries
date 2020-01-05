
# ------------------------- MODIFY VALUES BELOW ----------------------------------------

set(CPP_ARGS "")
set(C_ARGS   "")

if(WIN32) # ----------------------------------------------------------------------------
    set(WINDOWS_SDK "10.0.17763.0")
    set(CMAKE_SYSTEM_VERSION ${WINDOWS_SDK})
    set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION ${WINDOWS_SDK})

    SET(INSTALL_PREFIX "C:/Source/CMake/Radicalware")

    set(CPP_ARGS " /EHsc")
    set(C_ARGS   CPP_ARGS)
    set(CPP_ARGS "${CPP_ARGS} /std:c++17")

    set(CMAKE_PATH "C:/Program Files/CMake/share/cmake-3.16/Modules")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_PATH})
    list(APPEND CMAKE_MODULE_PATH ${RADICAL_PATH})
else() # -----------------------------------------------------------------------------
    SET(INSTALL_PREFIX "/opt/Radicalware")

    set(CPP_ARGS " -Wfatal-errors -finput-charset=UTF-8 -fPIC -pthread")
    set(CPP_ARGS "${CPP_ARGS} -Wno-unused-variable")
    set(C_ARGS   CPP_ARGS)
    set(CPP_ARGS "${CPP_ARGS} -std=c++17")

    set(CMAKE_PATH "/usr/share/cmake-3.16/Modules")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_PATH})
    list(APPEND CMAKE_MODULE_PATH ${RADICAL_PATH})
    list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-3.16/Utilities/cmlibarchive/build/cmake")
endif() # ----------------------------------------------------------------------------

# ------------------------- MODIFY VALUES ABOVE --------------------------------------

# --------------- DON'T MODIFY (CALCULATED) ------------------------------------------

if(BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
else()
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
endif()

if(BUILD_TYPE STREQUAL "Release")
    set(${CPP_ARGS} " ${CPP_ARGS} -O2")
    set(C_ARGS      " ${C_ARGS}   -O2")

elseif(NOT WIN32) # and debug
    set(${CPP_ARGS} " ${CPP_ARGS} -g3 -ggdb ")
    set(${C_ARGS}   " ${C_ARGS}   -g3 -ggdb ")
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set(CMAKE_CXX_FLAGS ${CPP_ARGS})
#set(CMAKE_C_FLAGS   ${C_ARGS})

set(EXT_HEADER_PATH ${INSTALL_PREFIX}/Libraries/Headers)
set(EXT_BIN_PATH    ${INSTALL_PREFIX}/Libraries/Build/${BUILD_TYPE})

set(PROJECT_DIR ${INSTALL_PREFIX}/Libraries/Projects)


MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY "${curdir}/${child}")
            SUBDIRLIST(NEXT_DIR "${curdir}/${child}")
            LIST(APPEND ${result} ${NEXT_DIR})
        ELSE()
            LIST(APPEND ${result} ${curdir}/${child})
        ENDIF()
    ENDFOREACH()
    list(REMOVE_DUPLICATES  ${result})
ENDMACRO()


function(CONFIGURE_VISUAL_STUDIO)
    
    foreach(SOURCE_FILE IN ITEMS ${ARGN})
        get_filename_component(SOURCE_PATH "${SOURCE_FILE}" PATH)
        string(REGEX REPLACE "${CMAKE_SOURCE_DIR}/[^/]+" "" RELATIVE_PATH "${SOURCE_PATH}")
        string(REPLACE "/" "\\" RELATIVE_PATH "${RELATIVE_PATH}")
        source_group("${RELATIVE_PATH}" FILES "${SOURCE_FILE}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(CONFIGURE_VISUAL_STUDIO)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------- DON'T MODIFY (CALCULATED) ------------------------------------------
