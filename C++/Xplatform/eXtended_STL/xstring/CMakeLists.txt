cmake_minimum_required(VERSION 3.12)

set(THIS xstring)

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

    ${INC}/std_xstring.h
    ${SRC}/std_xstring.cpp

    ${INC}/Color.h
    ${SRC}/Color.cpp
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
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------