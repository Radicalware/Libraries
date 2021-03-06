﻿cmake_minimum_required(VERSION 3.16)

set(LIB xstring)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PRIVATE_LIB_LST ${LIB})

set(XSTRING_DIR ${PROJECT_DIR}/${LIB})
set(INC         ${XSTRING_DIR}/include)
set(SRC         ${XSTRING_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} STATIC ${PROJECT_FILES})
add_library(Radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PRIVATE
    
    ${NEXUS_DIR}/include
    ${XVECTOR_DIR}/include
    ${XSTRING_DIR}/include
)

target_link_libraries(${LIB} Radical::Nexus)
target_link_libraries(${LIB} Radical::xvector)

target_link_libraries(${LIB} Radical_Mod::re2)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
