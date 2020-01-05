cmake_minimum_required(VERSION 3.12)

set(LIB mc)
list(APPEND STATIC_LIB_LST ${LIB})

# -------------------------- CONFIGURATION ------------------------------------
set(MC_DIR  ${PROJECT_DIR}/${LIB})
set(INC     ${MC_DIR}/include)
set(SRC     ${MC_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC
	
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${MC_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
