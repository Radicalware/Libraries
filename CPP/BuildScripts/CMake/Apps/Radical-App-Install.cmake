
# Install the Executable
if(${debug} OR ${BuildAll})
    foreach(EXE ${EXES})
        if(${release})
            install(TARGETS ${EXE} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE})
        else()
            install(TARGETS ${EXE} DESTINATION ${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin)
        endif()
    endforeach()
    
    if(WIN32)
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}")
    else()
        set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}")    
    endif()

    foreach(SharedLib ${SharedLibs})
        install(
            FILES "${OUTPUT_DIR}/bin/${PF}${SharedLib}.${SH}"
            CONFIGURATIONS ${BUILD_TYPE}
            DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin"
            OPTIONAL
        )
        if(${IsWindows})
            install(
                FILES "${OUTPUT_DIR}/lib/${PF}${SharedLib}.${ST}"
                CONFIGURATIONS ${BUILD_TYPE}
                DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
                OPTIONAL
            )
        endif()
    endforeach()

    foreach(StaticLib ${StaticLibs})
        install(
            FILES "${OUTPUT_DIR}/lib/${PF}${StaticLib}.${ST}"
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

