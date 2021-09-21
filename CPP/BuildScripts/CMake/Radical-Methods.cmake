﻿macro(PrintBanner)
    message("")
    message("---------------------------------------------------------------------------------------------------")
    message("THIS:              ${THIS}")
    message("BUILD_TYPE:        ${BUILD_TYPE}")
    message("RADICAL_BASE       ${RADICAL_BASE}")
    message("INSTALL_PREFIX:    ${INSTALL_PREFIX}")
    message("EXT_HEADER_PATH:   ${EXT_HEADER_PATH}")
    message("EXT_BIN_PATH:      ${EXT_BIN_PATH}")
    message("CMAKE_PATH:        ${CMAKE_PATH}")
    message("RADICAL_PATH:      ${RADICAL_PATH}")
    if(WIN32)
        message("MSVC SDK:          ${WINDOWS_SDK}")
    endif()
    message("---------------------------------------------------------------------------------------------------")
endmacro()


function(PrintList List) # pass in no brace or quotes
    foreach(Item IN LISTS ${List})
        message(">> ${Item}") # can't use VAR
    endforeach()
endfunction(PrintList)

function(SetAllDependenciesOn Target)
    if(${BuildAll})
        foreach(Lib ${StaticLibs})
            add_dependencies(${Target} ${Lib})
        endforeach()
        foreach(Lib ${SharedLibs})
            add_dependencies(${Target} ${Lib})
        endforeach()
    endif()
endfunction()

# recursivly find all our needed Project/Solution files
MACRO(FindProgramFiles RESULT CUR_DIR)
    FILE(GLOB CHILDREN RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD ${CHILDREN})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            FindProgramFiles(${RESULT} "${CUR_DIR}/${CHILD}")
        ELSE()
            LIST(APPEND ${RESULT} "${CUR_DIR}/${CHILD}")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

MACRO(FindProgramDirs RESULT CUR_DIR)
    FILE(GLOB DIR_LST RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD IN ITEMS ${DIR_LST})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            LIST(APPEND ${RESULT} "${INSTALL_DIR}/${CHILD}")
            FindProgramDirs(${RESULT} "${CUR_DIR}/${CHILD}")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

MACRO(FindIncludeDirs RESULT CUR_DIR)
    FILE(GLOB DIR_LST RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD IN ITEMS ${DIR_LST})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            LIST(APPEND ${RESULT} "${INSTALL_DIR}/${CHILD}/include")
        ENDIF()
    ENDFOREACH()
ENDMACRO()


macro(SetLocalInstallDirs)
    set(LocalInstalldir "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}")
    if(WIN32)
        SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${LocalInstalldir}/bin")
        SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${LocalInstalldir}/bin")

        SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${LocalInstalldir}/bin")
        SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LocalInstalldir}/bin")

        SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${LocalInstalldir}/lib")
        SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${LocalInstalldir}/lib")
    else()
        set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LocalInstalldir}")
        set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LocalInstalldir}/bin")
        set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LocalInstalldir}/lib")
    endif()
endmacro()

function(post_build_copy target_name file_name from_dir to_dir)
    add_custom_command(
        TARGET "${target_name}" POST_BUILD # PRE_BUILD, PRE_LINK, POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${from_dir}/${file_name}"
                "${to_dir}/${file_name}"
        DEPENDS "${from_dir}/${file_name}"
    )
endfunction()



# DEBUG MESSAGES
# MESSAGE("==================================================================")
# MESSAGE("FULL_PATH        = ${FULL_PATH}")
# MESSAGE("SOURCE_PATH      = ${SOURCE_PATH}")
# MESSAGE("FILE_NAME        = ${FILE_NAME}")
# MESSAGE("BASE             = ${BASE}")

function(SetVisualStudioFilters BreakPoint Files)
    foreach(REL_PATH IN ITEMS ${Files})
        get_filename_component(FULL_PATH "${REL_PATH}" ABSOLUTE)

        if(NOT(FULL_PATH MATCHES  ".*${BreakPoint}.*"))
            continue()
        endif()

        # FULL_PATH is the full path of the file's install location (not the solution/project location)
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${FULL_PATH}" FULL_LEN)

        string(REGEX REPLACE "${BreakPoint}.*" "" NO_SOLUTION ${FULL_PATH})
        string(LENGTH   "${NO_SOLUTION}/${BreakPoint}" SLN_FOLDER_LEN)
        string(LENGTH   "${FILE_NAME}" FILE_LEN)

        math(EXPR RELATIVE_LEN   "${FULL_LEN}  - ${SLN_FOLDER_LEN} - ${FILE_LEN}")
        string(SUBSTRING  ${FULL_PATH} ${SLN_FOLDER_LEN} ${RELATIVE_LEN} RELATIVE_PATH)
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}")

        # message("FULL_PATH     = ${FULL_PATH}")
        # message("Solution Path = ${CMAKE_SOURCE_DIR}/Solution/")
        # message("RELATIVE_PATH = ${RELATIVE_PATH}")
        # message("--")

        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TargetProject})
endfunction()

macro(install_symlink filepath sympath) # Windows requires admin right to create sym links
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
endmacro(install_symlink)

