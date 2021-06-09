
# -------------------------- INSTALL ------------------------------------------

# Install the Executable

if(${release})
    install(TARGETS ${THIS} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE})
else()
    install(TARGETS ${THIS} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin)
endif()


if(${build_all})
    # Install to Applications
    install( 
        TARGETS       ${SHARED_LIB_LST} 
        DESTINATION   ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
    )
    install(
        TARGETS       ${STATIC_LIB_LST}  
        DESTINATION   ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
    )

    # Install to Libraries
    install(
        TARGETS       ${SHARED_LIB_LST}
        DESTINATION   ${INSTALL_PREFIX}/../Libraries/Build/${BUILD_TYPE}/bin
    )
    install(
        TARGETS       ${STATIC_LIB_LST}
        DESTINATION   ${INSTALL_PREFIX}/../Libraries/Build/${BUILD_TYPE}/lib
    )
endif()

# Header/Src Files
install (   
    DIRECTORY   "${BUILD_DIR}/include"
    DESTINATION "${INSTALL_PREFIX}/Solutions/${THIS}"
)
install (   
    DIRECTORY   "${BUILD_DIR}/src"
    DESTINATION "${INSTALL_PREFIX}/Solutions/${THIS}"
)

# -------------------------- INSTALL ------------------------------------------


