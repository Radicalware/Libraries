cmake_minimum_required(VERSION 3.16)

set(LIB Nexus)
list(APPEND STATIC_LIB_LST ${LIB})

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PRIVATE_LIB_LST ${LIB})

set(NEXUS_DIR  ${PROJECT_DIR}/${LIB})
set(INC        ${NEXUS_DIR}/include)
set(SRC        ${NEXUS_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} STATIC ${PROJECT_FILES})
add_library(Radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PRIVATE

    ${NEXUS_DIR}/include
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
