cmake_minimum_required(VERSION 3.10)

set(LIB ac)

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    if("${BUILD_TYPE}" STREQUAL "Release")
        add_definitions( "-O2" )
    endif()
else()
    set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -fPIC")
    if("${BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(LIB_DIR  ${INSTALL_PREFIX}/Projects/${LIB})
set(INC      ${LIB_DIR}/include)
set(SRC      ${LIB_DIR}/src)

set(AC_DIR   ${INSTALL_PREFIX}/Projects/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC
	
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${AC_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
