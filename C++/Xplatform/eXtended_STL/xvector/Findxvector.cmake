cmake_minimum_required(VERSION 3.12)

set(LIB xvector)

# -------------------------- CONFIGURATION ------------------------------------
set(XVECTOR_DIR  ${PROJECT_DIR}/${LIB})
set(INC          ${XVECTOR_DIR}/include)
set(SRC          ${XVECTOR_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB}  STATIC
	
        ${INC}/${LIB}.h
        ${SRC}/${LIB}.cpp

        # -------------------------------
        
        ${INC}/base_val_${LIB}.h
        ${SRC}/base_val_${LIB}.cpp

        ${INC}/base_ptr_${LIB}.h
        ${SRC}/base_ptr_${LIB}.cpp

        # -------------------------------

        ${INC}/val_obj_xvector.h
        ${SRC}/val_obj_xvector.cpp

        ${INC}/val_prim_xvector.h
        ${SRC}/val_prim_xvector.cpp

        ${INC}/ptr_obj_xvector.h
        ${SRC}/ptr_obj_xvector.cpp

        ${INC}/ptr_prim_xvector.h
        ${SRC}/ptr_prim_xvector.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${NEXUS_DIR}/include
        ${XVECTOR_DIR}
        ${XVECTOR_DIR}/include
)

target_link_libraries(${LIB} radical::Nexus)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
