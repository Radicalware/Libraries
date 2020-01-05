

if(WIN32)
    # set paths are the default locations of the cmake_install.cmake configuration by pcre2-10.34

    # Copyright (C) 2007-2009 LuaDist. (For the Windows Install Method)
    # Created by Peter Kapec <kapecp@gmail.com>
    # Modified bo Joel Leagues

    set(PCRE_INCLUDE_DIR "C:/Program Files (x86)/PCRE/include")
    set(PCRE_BIN_DIR "C:/Program Files (x86)/PCRE/bin")
    set(PCRE_LIB_DIR "C:/Program Files (x86)/PCRE/lib")

    FIND_PATH(${PCRE_INCLUDE_DIR} NAMES pcre.h)
    FIND_LIBRARY(${PCRE_LIB_DIR} NAMES pcre)

    # Handle the QUIETLY and REQUIRED arguments and set PCRE_FOUND to TRUE if all listed variables are TRUE.
    INCLUDE(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(PCRE DEFAULT_MSG PCRE_LIB_DIR PCRE_INCLUDE_DIR)

    # Copy the results to the output variables.
    IF(PCRE_FOUND)
        SET(PCRE_LIBRARIES ${PCRE_LIB_DIR})
        SET(PCRE_INCLUDE_DIRS ${PCRE_INCLUDE_DIR})
    ELSE(PCRE_FOUND)
        SET(PCRE_LIBRARIES)
        SET(PCRE_INCLUDE_DIRS)
    ENDIF(PCRE_FOUND)

    MARK_AS_ADVANCED(PCRE_INCLUDE_DIRS PCRE_LIBRARIES)

    target_include_directories(${THIS} PRIVATE ${PCRE_INCLUDE_DIR})
    target_link_libraries(${THIS} PRIVATE "${PCRE_LIB_DIR}/pcre.lib")
else()
    find_package(PCREPOSIX)
    target_include_directories(${THIS} PRIVATE ${PCRE_INCLUDE_DIR})
    target_link_libraries(${THIS} PRIVATE ${PCRE_LIBRARIES})

endif()