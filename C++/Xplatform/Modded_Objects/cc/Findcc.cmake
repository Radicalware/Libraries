cmake_minimum_required(VERSION 3.12)

set(LIB cc)

# -------------------------- CONFIGURATION ------------------------------------
set(CC_DIR  ${PROJECT_DIR}/${LIB})
set(INC     ${CC_DIR}/include)
set(SRC     ${CC_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} SHARED 

    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp
)
add_library(radical_mod::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${CC_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
