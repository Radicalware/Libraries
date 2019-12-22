cmake_minimum_required(VERSION 3.12)

set(THIS xvector)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

add_library( ${THIS} STATIC

        ${INC}/${THIS}.h
        ${SRC}/${THIS}.cpp

        # -------------------------------
        
        ${INC}/base_val_${THIS}.h
        ${SRC}/base_val_${THIS}.cpp

        ${INC}/base_ptr_${THIS}.h
        ${SRC}/base_ptr_${THIS}.cpp

        # -------------------------------

        ${INC}/val_obj_xvector.h
        ${SRC}/val_obj_xvector.cpp

        ${INC}/val_prim_xvector.h
        ${SRC}/val_prim_xvector.cpp

        ${INC}/ptr_obj_xvector.h
        ${SRC}/ptr_obj_xvector.cpp

        ${INC}/ptr_prim_xvector.h
        ${SRC}/ptr_prim_xvector.cpp
)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE
        ${EXT_HEADER_PATH}
        ${INC}
        ${INC}/include
        ${EXT_HEADER_PATH}
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${SF})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------