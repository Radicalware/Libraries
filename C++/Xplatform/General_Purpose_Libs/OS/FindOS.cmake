cmake_minimum_required(VERSION 3.12)

set(LIB OS)

# -------------------------- CONFIGURATION ------------------------------------
set(OS_DIR  ${PROJECT_DIR}/${LIB})
set(INC     ${OS_DIR}/include)
set(SRC     ${OS_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------

add_library(${LIB} SHARED  

    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp

    ${INC}/dir_support/Dir_Type.h
    ${SRC}/dir_support/Dir_Type.cpp

    ${INC}/handlers/File.h
    ${SRC}/handlers/File.cpp

    ${INC}/handlers/CMD.h
    ${SRC}/handlers/CMD.cpp
)

add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PUBLIC
        ${NEXUS_DIR}/include
        ${XVECTOR_DIR}/include
        ${XSTRING_DIR}/include
        ${XMAP_DIR}/include

        ${OS_DIR}/include
        ${OS_DIR}/include/dir_support
        ${OS_DIR}/include/handlers
)

target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
