
# -------------------------- INSTALL ------------------------------------------

# Install the Executable
install(TARGETS ${THIS} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE})

# Install to Applications
install( 
    TARGETS       ${SHARED_LIB_LST} 
    DESTINATION   ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
)
install(
    TARGETS       ${STATIC_LIB_LST}  
    DESTINATION   ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
)

# Install to Librariess
install(
    TARGETS ${SHARED_LIB_LST}
    DESTINATION   ${INSTALL_PREFIX}/../Libraries/Build/${BUILD_TYPE}/bin
)
install(
    TARGETS ${STATIC_LIB_LST}
    DESTINATION   ${INSTALL_PREFIX}/../Libraries/Build/${BUILD_TYPE}/lib
)

# Header/Src Files
install (   
    DIRECTORY           "${BUILD_DIR}/include"
    DESTINATION         "${INSTALL_PREFIX}/Solutions/${THIS}"
)
install (   
    DIRECTORY           "${BUILD_DIR}/src"
    DESTINATION         "${INSTALL_PREFIX}/Solutions/${THIS}"
)

# -------------------------- INSTALL ------------------------------------------


