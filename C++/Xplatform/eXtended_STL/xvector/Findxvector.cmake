cmake_minimum_required(VERSION 3.10)

set(LIB xvector)

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    if("${BUILD_TYPE}" STREQUAL "Release")
        add_definitions( "-O2" )
    endif()
else()
    set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -fPIC -pthread")
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
set(XVECTOR_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} 
    STATIC
        ${XVECTOR_DIR}/include/${LIB}.h
        ${XVECTOR_DIR}/src/${LIB}.cpp

        ${XVECTOR_DIR}/include/ptr_${LIB}.h
        ${XVECTOR_DIR}/src/ptr_${LIB}.cpp

        ${XVECTOR_DIR}/include/val_${LIB}.h
        ${XVECTOR_DIR}/src/val_${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${XVECTOR_DIR}/include
)

target_link_libraries(${LIB} radical::Nexus)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
