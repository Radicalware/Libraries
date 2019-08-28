cmake_minimum_required(VERSION 3.10)

set(LIB OS)

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
set(OS_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------

add_library(${LIB} 
    SHARED  
        ${OS_DIR}/include/${LIB}.h
        ${OS_DIR}/src/${LIB}.cpp

        ${OS_DIR}/include/dir_support/Dir_Type.h
        ${OS_DIR}/src/dir_support/Dir_Type.cpp

        ${OS_DIR}/include/dir_support/File_Names.h
        ${OS_DIR}/src/dir_support/File_Names.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        ${XMAP_DIR}/include

        ${OS_DIR}/include
        ${OS_DIR}/include/support_dir
)

target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
