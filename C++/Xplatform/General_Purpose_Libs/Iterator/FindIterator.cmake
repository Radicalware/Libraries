cmake_minimum_required(VERSION 3.16)

set(LIB Iterator)
list(APPEND SHARED_LIB_LST ${LIB})

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PUBLIC_LIB_LST ${LIB})

set(ITER_DIR ${PROJECT_DIR}/${LIB})
set(INC      ${ITER_DIR}/include)
set(SRC      ${ITER_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} SHARED ${PROJECT_FILES})
add_library(Radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB} PUBLIC
    
    ${ITER_DIR}/include
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
