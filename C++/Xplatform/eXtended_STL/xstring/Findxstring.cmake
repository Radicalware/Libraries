cmake_minimum_required(VERSION 3.12)

set(LIB xstring)

# -------------------------- CONFIGURATION ------------------------------------
set(XSTRING_DIR ${PROJECT_DIR}/${LIB})
set(INC         ${XSTRING_DIR}/include)
set(SRC         ${XSTRING_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC 

    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp

    ${INC}/std_xstring.h
    ${SRC}/std_xstring.cpp

    ${INC}/Color.h
    ${SRC}/Color.cpp
)
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${NEXUS_DIR}/include
        ${XVECTOR_DIR}/include
        
        ${XSTRING_DIR}/include
)

target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
