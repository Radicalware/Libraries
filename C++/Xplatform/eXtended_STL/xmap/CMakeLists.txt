cmake_minimum_required(VERSION 3.12)

set(THIS xmap)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

add_library(${THIS} STATIC 

    ${INC}/${THIS}.h
    ${SRC}/${THIS}.cpp

    ${INC}/val2_xmap.h
    ${SRC}/val2_xmap.cpp

    ${INC}/ptr2_xmap.h
    ${SRC}/ptr2_xmap.cpp

    ${INC}/ptr_val_xmap.h
    ${SRC}/ptr_val_xmap.cpp
    
    ${INC}/val_ptr_xmap.h
    ${SRC}/val_ptr_xmap.cpp

)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_HEADER_PATH}
        ${INC}
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xvector${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xstring${SF})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------