

# -------------------------- INSTALL ----------------------------------------------------

# Binary File

FILE(REMOVE "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin/lib${THIS}${SH}")
FILE(REMOVE "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib/lib${THIS}${ST}")

if(WIN32)
    install(
        TARGETS ${THIS}
            RUNTIME DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
            ARCHIVE DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
        CONFIGURATIONS ${BUILD_TYPE}
    )
else()
    install(
        TARGETS ${THIS}
            RUNTIME DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            LIBRARY DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin
            ARCHIVE DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib
        CONFIGURATIONS ${BUILD_TYPE}
    )
endif()

# Include file
install (   
    DIRECTORY       ${BUILD_DIR}/include/
    DESTINATION     ${EXT_HEADER_PATH}
    CONFIGURATIONS  ${BUILD_TYPE}
)

# Header/Src Files
install (   
    DIRECTORY       ${BUILD_DIR}/include
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
    CONFIGURATIONS  ${BUILD_TYPE}
)
install (   
    DIRECTORY       ${BUILD_DIR}/src 
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
    CONFIGURATIONS  ${BUILD_TYPE}
)

# CMake Module
install (   
    FILES           Find${THIS}.cmake 
    DESTINATION     ${RADICAL_PATH}
    CONFIGURATIONS  ${BUILD_TYPE}
)

# -------------------------- INSTALL ----------------------------------------------------
