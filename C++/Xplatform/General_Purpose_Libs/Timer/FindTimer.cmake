cmake_minimum_required(VERSION 3.12)

set(LIB Timer)
list(APPEND SHARED_LIB_LST ${LIB})

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PUBLIC_LIB_LST ${LIB})

set(TIMER_DIR  ${PROJECT_DIR}/${LIB})
set(INC        ${TIMER_DIR}/include)
set(SRC        ${TIMER_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")


add_library(${LIB} SHARED ${PROJECT_FILES})
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PUBLIC

    ${NEXUS_DIR}/include
    ${XVECTOR_DIR}/include
    ${XSTRING_DIR}/include
    ${XMAP_DIR}/include
    ${TIMER_DIR}/include
)

target_link_libraries(${LIB} radical_mod::re2)
target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
target_link_libraries(${LIB} radical::xmap)

target_link_libraries(${THIS} PRIVATE radical::${LIB})

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
