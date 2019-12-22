cmake_minimum_required(VERSION 3.12)

set(LIB Iterator)

# -------------------------- CONFIGURATION ------------------------------------
set(RE_DIR  ${PROJECT_DIR}/${LIB})
set(INC     ${RE_DIR}/include)
set(SRC     ${RE_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} SHARED
	
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${RE_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
