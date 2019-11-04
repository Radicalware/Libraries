
cmake_minimum_required(VERSION 3.10)

set(THIS xvector)
project (${THIS})

message("")
message("--------------------------------------------------------------------------------")
message("THIS:              ${THIS}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("INSTALL_PREFIX:    ${INSTALL_PREFIX}")
message("EXT_INCLUDE_PATH:  ${EXT_INCLUDE_PATH}")
message("EXT_BIN_PATH:      ${EXT_BIN_PATH}")
message("MODULE_PATH:       ${MODULE_PATH}")
message("--------------------------------------------------------------------------------")

# -------------------------- ARGUMENTS ----------------------------------------
set(CMAKE_BUILD_TYPE "${BUILD_TYPE}")
if(MSVC)
    set(PF "")
    set(SF ".lib")

    if("${BUILD_TYPE}" STREQUAL "Release")
        add_definitions( "-O2" )
    endif()
else() # NIX
    set(PF "lib")
    set(SF ".a")
    set(LINUX_ARGS "-std=c++17 -Wfatal-errors -finput-charset=UTF-8 -fPIC -pthread")

    if("${BUILD_TYPE}" STREQUAL "Release")
        message("Buidling with -O2 ${BUILD_TYPE}")
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -O2")
        set(CMAKE_C_FLAGS "-O2")
    else()
        set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT}  ${LINUX_ARGS} -g2")
        set(CMAKE_C_FLAGS "-g2")
    endif()
endif()
# -------------------------- ARGUMENTS ----------------------------------------
# -------------------------- CONFIGURATION ------------------------------------
set(LIB ${THIS})
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Project)
set(INCLUDE   ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

set(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INCLUDE}) 

if(EXISTS "${PROJECT_SOURCE_DIR}/Find${THIS}.cmake")
    file(REMOVE ${MODULE_PATH}/Find${THIS}.cmake)
else()
    message(FATAL_ERROR "/n/n!!!! Find${THIS}.cmake NOT FOUND !!!!/n/n")
endif()
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library( ${THIS} STATIC

        ${INCLUDE}/base_val_${LIB}.h
        ${SRC}/base_ptr_${LIB}.cpp

        ${INCLUDE}/val_obj_xvector.h
        ${SRC}/val_obj_xvector.cpp

        ${INCLUDE}/val_prim_xvector.h
        ${SRC}/val_prim_xvector.cpp
        # -------------------------------------
        ${INCLUDE}/base_ptr_xvector.h
        ${SRC}/base_ptr_xvector.cpp

        ${INCLUDE}/ptr_obj_xvector.h
        ${SRC}/ptr_obj_xvector.cpp

        ${INCLUDE}/ptr_prim_xvector.h
        ${SRC}/ptr_prim_xvector.cpp
)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_INCLUDE_PATH}
        ${INCLUDE}
)

# Required Libs ---------------------------------------------------------------
message ("No Required Libs Required")
# -------------------------- BUILD --------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/${BUILD_TYPE}/lib/${PF}Nexus${SF})
# -------------------------- INSTALL ------------------------------------------

# Binary File
install(TARGETS ${THIS}
        RUNTIME DESTINATION build/${BUILD_TYPE}/bin
        LIBRARY DESTINATION build/${BUILD_TYPE}/lib
        ARCHIVE DESTINATION build/${BUILD_TYPE}/lib
)

# Include file
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         .
)

# Header/Src Files
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         code/Projects/${THIS}
)
install (   
    DIRECTORY           ${BUILD_DIR}/src 
    DESTINATION         code/Projects/${THIS}
)

# CMake Module
if(UNIX)
    install (   
        FILES           Find${THIS}.cmake 
        DESTINATION     /${PART_MODULE_PATH}
    ) 
else()
    install (   
        FILES           Find${THIS}.cmake 
        DESTINATION     C:/${PART_MODULE_PATH}
    )
endif()
# -------------------------- INSTALL ------------------------------------------
# -------------------------- END ----------------------------------------------
