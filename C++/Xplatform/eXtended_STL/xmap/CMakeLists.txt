
cmake_minimum_required(VERSION 3.10)

set(LIB xmap)
project (${LIB})

message("")
message("--------------------------------------------------------------------------------")
message("LIB:               ${LIB}")
message("BUILD_TYPE:        ${BUILD_TYPE}")
message("INSTALL_PREFIX:    ${INSTALL_PREFIX}")
message("EXT_HEADER_PATH:   ${EXT_HEADER_PATH}")
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
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/Project)
set(INC       ${BUILD_DIR}/include)
set(SRC       ${BUILD_DIR}/src)

set(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "" FORCE)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${INC}) 

if(EXISTS "${PROJECT_SOURCE_DIR}/Find${LIB}.cmake")
    file(REMOVE ${MODULE_PATH}/Find${LIB}.cmake)
else()
    message(FATAL_ERROR "/n/n!!!! Find${LIB}.cmake NOT FOUND !!!!/n/n")
endif()
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------
add_library(${LIB} STATIC 

    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp

    ${INC}/val2_xmap.h
    ${SRC}/val2_xmap.cpp

    ${INC}/ptr2_xmap.h
    ${SRC}/ptr2_xmap.cpp

    ${INC}/ptr_val_xmap.h
    ${SRC}/ptr_val_xmap.cpp
    
    ${INC}/val_ptr_xmap.h
    ${SRC}/val_ptr_xmap.cpp

)
add_library(radical::${LIB} ALIAS ${LIB})

target_include_directories(${LIB}
    PRIVATE
         ${EXT_HEADER_PATH}
        
)

# Required Libs ---------------------------------------------------------------
target_link_libraries(${LIB} ${EXT_BIN_PATH}/${BUILD_TYPE}/lib/${PF}Nexus${SF})
target_link_libraries(${LIB} ${EXT_BIN_PATH}/${BUILD_TYPE}/lib/${PF}xvector${SF})
target_link_libraries(${LIB} ${EXT_BIN_PATH}/${BUILD_TYPE}/lib/${PF}xstring${SF})
# -------------------------- BUILD --------------------------------------------
# -------------------------- INSTALL ------------------------------------------

# Binary File
if(msvc)
    install(TARGETS ${LIB}
            RUNTIME DESTINATION Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION Build/${BUILD_TYPE}/lib
            ARCHIVE DESTINATION Build/${BUILD_TYPE}/lib
    )
else()
    install(TARGETS ${LIB}
            RUNTIME DESTINATION Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION Build/${BUILD_TYPE}/bin
            ARCHIVE DESTINATION Build/${BUILD_TYPE}/lib
    )
endif()

# Include file
install (   
    DIRECTORY           ${BUILD_DIR}/include/
    DESTINATION         ${EXT_HEADER_PATH}/
)

# Header/Src Files
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         Projects/${LIB}
)
install (   
    DIRECTORY           ${BUILD_DIR}/src 
    DESTINATION         Projects/${LIB}
)

# CMake Module
if(UNIX)
    install (   
        FILES           Find${LIB}.cmake 
        DESTINATION     /${PART_MODULE_PATH}
    ) 
else()
    install (   
        FILES           Find${LIB}.cmake 
        DESTINATION     C:/${PART_MODULE_PATH}
    )
endif()

# -------------------------- INSTALL ------------------------------------------
# -------------------------- END ----------------------------------------------
