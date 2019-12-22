cmake_minimum_required(VERSION 3.12)

set(THIS Timer)

# -------------------------- CONFIG -----------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

# -------------------------- BUILD ------------------------------------------------------

project (${THIS})

add_library(${THIS} SHARED 

    ${INC}/${THIS}.h
    ${SRC}/${THIS}.cpp
)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PUBLIC
        ${EXT_HEADER_PATH}
        ${INC}
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xvector${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xstring${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xmap${SF})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------