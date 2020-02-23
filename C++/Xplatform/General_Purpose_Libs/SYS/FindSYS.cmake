cmake_minimum_required(VERSION 3.16)

set(LIB SYS)

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PUBLIC_LIB_LST ${LIB})

set(SYS_DIR  ${PROJECT_DIR}/${LIB})
set(INC      ${SYS_DIR}/include)
set(SRC      ${SYS_DIR}/src)
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} SHARED 
    
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)
add_library(Radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB} PUBLIC

    ${NEXUS_DIR}/include
    ${RE2_DIR}/include
    ${XVECTOR_DIR}/include
    ${XSTRING_DIR}/include
    ${XMAP_DIR}/include
    ${SYS_DIR}/include
)

target_link_libraries(${LIB} Radical::Nexus)
target_link_libraries(${LIB} Radical::xvector)
target_link_libraries(${LIB} Radical::xstring)
target_link_libraries(${LIB} Radical::xmap)

target_link_libraries(${LIB} Radical_Mod::re2)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
