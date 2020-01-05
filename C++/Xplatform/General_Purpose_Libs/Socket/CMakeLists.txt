cmake_minimum_required(VERSION 3.12)

set(THIS Socket)

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

    # --------------------------------
    ${INC}/${THIS}.h
    ${SRC}/${THIS}.cpp

    ${INC}/Buffer.h
    ${SRC}/Buffer.cpp

    ${INC}/NixNet.h 
    ${SRC}/NixNet.cpp
    # --------------------------------
    ${INC}/Server.h
    ${SRC}/Server.cpp

    ${INC}/Server/Win_Server.h
    ${SRC}/Server/Win_Server.cpp

    ${INC}/Server/Nix_Server.h
    ${SRC}/Server/Nix_Server.cpp
    # --------------------------------
    ${INC}/Client.h
    ${SRC}/Client.cpp

    ${INC}/Client/Win_Client.h
    ${SRC}/Client/Win_Client.cpp

    ${INC}/Client/Nix_Client.h
    ${SRC}/Client/Nix_Client.cpp
    # --------------------------------
)

add_library(radical::${THIS} ALIAS ${THIS})

target_include_directories(${THIS}
    PRIVATE 
        ${EXT_HEADER_PATH}
        ${INC}
        ${INC}/Client
        ${INC}/Server
)

# Required Libs -------------------------------------------------------------------------
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}Nexus${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xvector${SF})
target_link_libraries(${THIS} ${EXT_BIN_PATH}/lib/${PF}xstring${SF})
# -------------------------- INSTALL ----------------------------------------------------

include("${RADICAL_PATH}/Radical-Lib-Install.cmake")

# -------------------------- END --------------------------------------------------------