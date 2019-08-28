cmake_minimum_required(VERSION 3.10)

set(LIB re)

# -------------------------- ARGUMENTS ----------------------------------------
set(DBG_ARG "")
if("${BUILD_TYPE}" STREQUAL "Release")
	add_definitions( "-O2" )
else()
	if(UNIX)
		set(DBG_ARG "-g2")
		set(CMAKE_C_FLAGS ${DBG_ARG})
		set(CMAKE_BUILD_TYPE Debug)
	endif()
endif()

if(UNIX)
	set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -Wfatal-errors -fPIC ${DBG_ARG}")
	set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS}")
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(RE_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} 
    SHARED
        ${RE_DIR}/include/${LIB}.h
        ${RE_DIR}/src/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${RE_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
