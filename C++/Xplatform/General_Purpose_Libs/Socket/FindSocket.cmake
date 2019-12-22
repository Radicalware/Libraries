cmake_minimum_required(VERSION 3.12)

set(LIB Socket)

# -------------------------- CONFIGURATION ------------------------------------
set(SOCKET_DIR  ${PROJECT_DIR}/${LIB})
set(INC         ${SOCKET_DIR}/include)
set(SRC         ${SOCKET_DIR}/src)
# -------------------------- CONFIGURATION ------------------------------------
# -------------------------- BUILD --------------------------------------------

add_library(${LIB} STATIC  

    # --------------------------------
    ${INC}/${LIB}.h
    ${SRC}/${LIB}.cpp

    ${INC}/Buffer.h
    ${SRC}/Buffer.cpp
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

add_library(radical::${LIB} ALIAS ${LIB})

include_directories(${LIB}
    PRIVATE
        ${SOCKET_DIR}/include
        ${SOCKET_DIR}/include/Client
        ${SOCKET_DIR}/include/Server
)

target_link_libraries(${LIB} radical::Nexus)
target_link_libraries(${LIB} radical::xvector)
target_link_libraries(${LIB} radical::xstring)
# target_link_libraries(${LIB} radical::xmap)
# -------------------------- BUILD --------------------------------------------
# -------------------------- END ----------------------------------------------
