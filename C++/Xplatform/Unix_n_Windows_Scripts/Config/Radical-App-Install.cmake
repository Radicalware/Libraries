
# -------------------------- INSTALL ------------------------------------------

install(TARGETS ${THIS} DESTINATION ${INSTALL_PREFIX}/../Applications/Build/${BUILD_TYPE})

install( 
    TARGETS       OS cc 
    DESTINATION   ${INSTALL_PREFIX}/../Applications/Build/${BUILD_TYPE}/bin
)
install(
    TARGETS       Nexus xvector xstring xmap 
    DESTINATION   ${INSTALL_PREFIX}/../Applications/Build/${BUILD_TYPE}/lib
)

# Header/Src Files
install (   
    DIRECTORY           ${BUILD_DIR}/include 
    DESTINATION         ../Applications/Solutions/${THIS}
)
install (   
    DIRECTORY           ${BUILD_DIR}/src 
    DESTINATION         ../Applications/Solutions/${THIS}
)

# -------------------------- INSTALL ------------------------------------------
