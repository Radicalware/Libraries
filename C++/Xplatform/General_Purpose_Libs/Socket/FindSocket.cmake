cmake_minimum_required(VERSION 3.12)

set(LIB Socket)
list(APPEND STATIC_LIB_LST ${LIB})

# -------------------------- PRE-CONFIG ---------------------------------------
list(APPEND PRIVATE_LIB_LST ${LIB})

set(SOCKET_DIR  ${PROJECT_DIR}/${LIB})
set(INC         ${SOCKET_DIR}/include)
set(SRC         ${SOCKET_DIR}/src)
# -------------------------- BUILD --------------------------------------------

UNSET(PROJECT_FILES)
SUBDIRLIST(PROJECT_FILES "${PROJECT_DIR}/${LIB}")

add_library(${LIB} STATIC ${PROJECT_FILES})
add_library(radical::${LIB} ALIAS ${LIB})

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB} PRIVATE
    
    ${SOCKET_DIR}/include
)

target_link_libraries(${LIB} radical_mod::re2)
target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)

target_link_libraries(${THIS} PRIVATE radical::${LIB})

# -------------------------- POST-CONFIG --------------------------------------
CONFIGURE_VISUAL_STUDIO_PROJECT(${PROJECT_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# -------------------------- END ----------------------------------------------
