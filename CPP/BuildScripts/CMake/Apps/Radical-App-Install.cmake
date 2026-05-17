
# Install the Executable
if(${debug} OR ${BuildAll})
    foreach(LoEXE ${EXES})
        if(${release})
            install(TARGETS ${LoEXE} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE})
        else()
            install(TARGETS ${LoEXE} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin)
        endif()
    endforeach()

    foreach(LoSharedLib ${SharedLibs})
        install(
            FILES "${OUTPUT_DIR}/bin/${PF}${LoSharedLib}.${SH}"
            CONFIGURATIONS ${BUILD_TYPE}
            DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin"
            OPTIONAL
        )
        if(${IsWindows})
            install(
                FILES "${OUTPUT_DIR}/lib/${PF}${LoSharedLib}.${ST}"
                CONFIGURATIONS ${BUILD_TYPE}
                DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
                OPTIONAL
            )
        endif()
    endforeach()

    foreach(LoStaticLib ${StaticLibs})
        install(
            FILES "${OUTPUT_DIR}/lib/${PF}${LoStaticLib}.${ST}"
            CONFIGURATIONS ${BUILD_TYPE}
            DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
            OPTIONAL
        )
    endforeach()

    # Header/Src Files
    install (   
        DIRECTORY   "${BUILD_DIR}/include"
        DESTINATION "${INSTALL_PREFIX}/Solutions/${THIS}"
    )
    install (   
        DIRECTORY   "${BUILD_DIR}/src"
        DESTINATION "${INSTALL_PREFIX}/Solutions/${THIS}"
    )
endif()
