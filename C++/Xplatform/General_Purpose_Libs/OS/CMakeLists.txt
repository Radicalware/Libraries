cmake_minimum_required(VERSION 3.12)

set(THIS OS)

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

    ${INC}/dir_support/Dir_Type.h
    ${SRC}/dir_support/Dir_Type.cpp

    ${INC}/handlers/File.h
    ${SRC}/handlers/File.cpp

    ${INC}/handlers/CMD.h
    ${SRC}/handlers/CMD.cpp

)
add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PUBLIC 
        ${EXT_HEADER_PATH}
        ${INC}
        ${INC}/dir_support
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xvector${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xstring${SF})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------