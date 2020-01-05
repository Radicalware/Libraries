cmake_minimum_required(VERSION 3.12)

set(LIB Nexus)
list(APPEND STATIC_LIB_LST ${LIB})

# -------------------------- CONFIGURATION ------------------------------------
set(NEXUS_DIR  ${PROJECT_DIR}/${LIB})
set(INC        ${NEXUS_DIR}/include)
set(SRC        ${NEXUS_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC 

    ${SRC}/NX_Threads.cpp
    ${INC}/NX_Threads.h

    ${SRC}/NX_Mutex.cpp
    ${INC}/NX_Mutex.h

    ${SRC}/Task.cpp
    ${INC}/Task.h

    ${SRC}/Job.cpp
    ${INC}/Job.h

    ${SRC}/${LIB}.cpp
    ${INC}/${LIB}.h

    ${SRC}/${LIB}_void.cpp
    ${INC}/${LIB}_void.h

    ${SRC}/${LIB}_T.cpp
    ${INC}/${LIB}_T.h
)
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE        
        ${NEXUS_DIR}/include
)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
