﻿
set_target_properties(${THIS} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)

FILE(REMOVE "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin/${PF}${THIS}.${SH}")
FILE(REMOVE "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib/${PF}${THIS}.${ST}")
FILE(REMOVE "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib/${PF}${THIS}.${OBJ}")
FILE(REMOVE "${RADICAL_PATH}/Find${THIS}.cmake")

if(WIN32)
    set(OBJ_FILE_PATH "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${THIS}.dir/${BUILD_TYPE}/${THIS}.${OBJ}")
else()
    set(OBJ_FILE_PATH "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/CMakeFiles/${THIS}.dir/Project/src/${THIS}.${OBJ}")
endif()

# Install shared files
install(
    FILES "${OUTPUT_DIR}/bin/${PF}${THIS}.${SH}"
    CONFIGURATIONS ${BUILD_TYPE}
    DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin"
    OPTIONAL
)

# Install static files
install(
    FILES "${OUTPUT_DIR}/lib/${PF}${THIS}.${ST}"
    CONFIGURATIONS ${BUILD_TYPE}
    DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
    OPTIONAL
)

if(${BuildAll})
    foreach(SharedLib ${SharedLibs})
        install(
            FILES "${OUTPUT_DIR}/bin/${PF}${SharedLib}.${SH}"
            CONFIGURATIONS ${BUILD_TYPE}
            DESTINATION "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/bin"
            OPTIONAL
        )
        if(${IsWindows})
            install(
                FILES "${OUTPUT_DIR}/bin/${PF}${SharedLib}.${ST}"
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
endif()

# Install Include file
install (   
    DIRECTORY       ${BUILD_DIR}/include/
    CONFIGURATIONS  ${BUILD_TYPE}
    DESTINATION     ${EXT_HEADER_PATH}
)

# Install Header/Src Files
install (   
    DIRECTORY       ${BUILD_DIR}/include
    CONFIGURATIONS  ${BUILD_TYPE}
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
)
install (   
    DIRECTORY       ${BUILD_DIR}/src 
    CONFIGURATIONS  ${BUILD_TYPE}
    DESTINATION     ${INSTALL_PREFIX}/Projects/${THIS}
)

# Install CMake Module
install (   
    FILES           Find${THIS}.cmake 
    CONFIGURATIONS  ${BUILD_TYPE}
    DESTINATION     ${RADICAL_PATH}
)

if(${IsApp})
    # Install object files
    install(
        FILES ${OBJ_FILE_PATH}
        CONFIGURATIONS ${BUILD_TYPE}
        DESTINATION   "${INSTALL_PREFIX}/Build/${BUILD_TYPE}/lib"
    )
endif()