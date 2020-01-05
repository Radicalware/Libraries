cmake_minimum_required(VERSION 3.12)

set(LIB xmap)
list(APPEND STATIC_LIB_LST ${LIB})

# -------------------------- CONFIGURATION ------------------------------------
set(XMAP_DIR  ${PROJECT_DIR}/${LIB})
set(INC       ${XMAP_DIR}/include)
set(SRC       ${XMAP_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC 
	
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp

    ${INC}/val2_xmap.h
    ${SRC}/val2_xmap.cpp

    ${INC}/ptr2_xmap.h
    ${SRC}/ptr2_xmap.cpp

    ${INC}/ptr_val_xmap.h
    ${SRC}/ptr_val_xmap.cpp
    
    ${INC}/val_ptr_xmap.h
    ${SRC}/val_ptr_xmap.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${NEXUS_DIR}/include
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        
        ${XMAP_DIR}/include
)

target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
