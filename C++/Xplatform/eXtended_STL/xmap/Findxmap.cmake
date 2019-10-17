cmake_minimum_required(VERSION 3.10)

set(LIB xmap)

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling ${THIS} with -O2 ${BUILD_TYPE}")
        add_definitions( "-O2" )
    endif()
else()
    set(LINUX_ARGS "-std=c++17 -finput-charset=UTF-8 -fPIC")
    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling with -O2 ${BUILD_TYPE}")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(XMAP_DIR ${INSTALL_PREFIX}/code/${LIB})
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC 
    ${XMAP_DIR}/include/${LIB}.h
    ${XMAP_DIR}/src/${LIB}.cpp
	
    ${XMAP_DIR}/include/val2_xmap.h
    ${XMAP_DIR}/src/val2_xmap.cpp

    ${XMAP_DIR}/include/ptr2_xmap.h
    ${XMAP_DIR}/src/ptr2_xmap.cpp

    ${XMAP_DIR}/include/ptr_val_xmap.h
    ${XMAP_DIR}/src/ptr_val_xmap.cpp

    ${XMAP_DIR}/include/val_ptr_xmap.h
    ${XMAP_DIR}/src/val_ptr_xmap.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        
        ${XMAP_DIR}/include
)

target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
