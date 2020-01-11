
include("${RADICAL_PATH}/Radical-Static-Vars.cmake")
include("${RADICAL_PATH}/Radical-Functions.cmake")
include("${RADICAL_PATH}/Radical-Header.cmake")

# -------------------------- CONFIGURATION ------------------------------------

set(INSTALL_PREFIX "${INSTALL_PREFIX}/Libraries")

if(WIN32)
    set(PF "")     # Prefix
    set(ST ".lib") # STatic
    set(SH ".dll") # SHared
else() # NIX
    set(PF "lib")
    set(ST ".a")
    set(SH ".so")
endif()

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Project)
set(INC       ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

set(cmake_configuration_types ${build_type} cache string "" force)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INC}) 
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#if(EXISTS "${CMAKE_SOURCE_DIR}/Find${THIS}.cmake")
#    file(REMOVE ${CMAKE_MODULE_PATH}/Find${THIS}.cmake)
#else()
#    message(FATAL_ERROR "Cannot Find: '${CMAKE_SOURCE_DIR}/Find${THIS}.cmake'")
#endif()

# -------------------------- CONFIGURATION ------------------------------------