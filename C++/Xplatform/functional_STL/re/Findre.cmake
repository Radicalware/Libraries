cmake_minimum_required(VERSION 3.16)

set(LIB re)
list(APPEND SHARED_LIB_LST ${LIB})

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PUBLIC_LIB_LST ${LIB})

set(RE_DIR ${PROJECT_DIR}/${LIB})
set(INC    ${RE_DIR}/include)
set(SRC    ${RE_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} SHARED ${PROJECT_FILES})
add_library(Radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB} PUBLIC

    ${RE_DIR}/include
)

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
