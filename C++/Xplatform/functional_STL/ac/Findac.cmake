cmake_minimum_required(VERSION 3.12)

set(LIB ac)

# -------------------------- CONFIGURATION ------------------------------------
set(AC_DIR  ${PROJECT_DIR}/${LIB})
set(INC     ${AC_DIR}/include)
set(SRC     ${AC_DIR}/src)
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
