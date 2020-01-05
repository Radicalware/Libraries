cmake_minimum_required(VERSION 3.12)

set(LIB SYS)
list(APPEND SHARED_LIB_LST ${LIB})

# -------------------------- CONFIGURATION ------------------------------------
set(SYS_DIR  ${PROJECT_DIR}/${LIB})
set(INC      ${SYS_DIR}/include)
set(SRC      ${SYS_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} SHARED 
    
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)
add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${NEXUS_DIR}/include
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        ${XMAP_DIR}/include
        
        ${SYS_DIR}/include
)

target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
