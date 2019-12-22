

# -------------------------- INSTALL ----------------------------------------------------

# Binary File
if(WIN32)
    install(TARGETS ${THIS}
            RUNTIME DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
            ARCHIVE DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
    )
else()
    install(TARGETS ${THIS}
            RUNTIME DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            ARCHIVE DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
    )
endif()

# Include file
install (   
    DIRECTORY       ${BUILD_DIR}/include
    DESTINATION     ${EXT_HEADER_PATH}
)

# Header/Src Files
install (   
    DIRECTORY       ${BUILD_DIR}/include 
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
)
install (   
    DIRECTORY       ${BUILD_DIR}/src 
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
)

# CMake Module
install (   
    FILES           Find${THIS}.cmake 
    DESTINATION     ${RADICAL_PATH}
)

# -------------------------- INSTALL ----------------------------------------------------
