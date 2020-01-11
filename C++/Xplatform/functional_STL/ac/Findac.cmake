cmake_minimum_required(VERSION 3.12)

set(LIB ac)
list(APPEND STATIC_LIB_LST ${LIB})
# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PRIVATE_LIB_LST ${LIB})

set(AC_DIR ${PROJECT_DIR}/${LIB})
set(INC    ${AC_DIR}/include)
set(SRC    ${AC_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} STATIC ${PROJECT_FILES})
add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PRIVATE
    
    ${AC_DIR}/include
)

target_link_libraries(${THIS} PRIVATE radical::${LIB})

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
