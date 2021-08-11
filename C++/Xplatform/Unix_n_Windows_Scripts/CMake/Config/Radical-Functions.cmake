function(print_list)
    foreach(VALUE IN ITEMS ${ARGN})
        message(">> ${VALUE}") # can't use VAR
    endforeach()
endfunction(print_list)


# recursivly find all our needed Project/Solution files
MACRO(find_program_files RESULT CUR_DIR)
    FILE(GLOB CHILDREN RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD ${CHILDREN})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            find_program_files(${RESULT} "${CUR_DIR}/${CHILD}")
        ELSE()
            LIST(APPEND ${RESULT} "${CUR_DIR}/${CHILD}")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

MACRO(find_program_dirs RESULT CUR_DIR)
    FILE(GLOB DIR_LST RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD IN ITEMS ${DIR_LST})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            LIST(APPEND ${RESULT} "${INSTALL_DIR}/${CHILD}")
            find_program_dirs(${RESULT} "${CUR_DIR}/${CHILD}")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

MACRO(find_include_dirs RESULT CUR_DIR)
    FILE(GLOB DIR_LST RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD IN ITEMS ${DIR_LST})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            LIST(APPEND ${RESULT} "${INSTALL_DIR}/${CHILD}/include")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

function(link_static TARGET_FILE STATIC_LIB)
    target_link_libraries(${TARGET_FILE} ${EXT_BIN_PATH}/lib/${PF}${STATIC_LIB}.${ST})
endfunction()

# for this to run well in Visual Studio do the following...
# Tools > Options > Projects and Solutions > Build and Run > 
#    uncheck "Only build startup project and dependinces on run"
function(link_dynamic TARGET_FILE DYNAMIC_LIB)
    if(${debug} OR ${build_all})
        if(WIN32)
            target_link_libraries(${TARGET_FILE}  "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/lib/${PF}${DYNAMIC_LIB}.${ST}")
            link_libraries("Debug\\lib\\${DYNAMIC_LIB}")
        else()
            target_link_libraries(${TARGET_FILE}  "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/bin/${PF}${DYNAMIC_LIB}.${SH}")
        endif()
    else()
        if(WIN32) # windows injects the .lib into the .exe to locate the .dll file at runtime
            target_link_libraries(${TARGET_FILE} ${EXT_BIN_PATH}/lib/${PF}${DYNAMIC_LIB}.${ST})
            link_libraries("Release\\lib\\${DYNAMIC_LIB}")
        else()
            target_link_libraries(${TARGET_FILE} ${EXT_BIN_PATH}/bin/${PF}${DYNAMIC_LIB}.${SH})
        endif()
    endif()
endfunction()


function(post_build_copy target_name file_name from_dir to_dir)

    add_custom_command(
        TARGET "${target_name}" POST_BUILD # PRE_BUILD, PRE_LINK, POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${from_dir}/${file_name}"
                "${to_dir}/${file_name}"
        DEPENDS "${from_dir}/${file_name}"
    )
endfunction()

function(install_static_lib TARGET_FILE)

    set(LIB_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/lib")

    if(WIN32)
        set(OBJ_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${TARGET_FILE}.dir/${BUILD_TYPE}")
    else()
        set(OBJ_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/CMakeFiles/${TARGET_FILE}.dir/opt/Radicalware/Libraries/Projects/${TARGET_FILE}/src")
    endif()

    if(${release})
        set(lib_destination "../Libraries") # only release global dlls go to libraries
    else()
        set(lib_destination "Applications")
    endif()

    post_build_copy(
        "${TARGET_FILE}"
        "${PF}${TARGET_FILE}.${ST}"
        "${LIB_DIR}"
        "${INSTALL_PREFIX}/${lib_destination}/Build/${BUILD_TYPE}/lib"
    )

    post_build_copy(
        "${TARGET_FILE}"
        "${TARGET_FILE}.${OBJ}"
        "${OBJ_DIR}"
        "${INSTALL_PREFIX}/${lib_destination}/Build/${BUILD_TYPE}/lib"
    )
endfunction()

function(install_dynamic_lib TARGET_FILE)

    set(DLL_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/bin")
    if(WIN32)
        set(OBJ_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${TARGET_FILE}.dir/${BUILD_TYPE}")
    else()
        set(OBJ_DIR "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/CMakeFiles/${TARGET_FILE}.dir/opt/Radicalware/Libraries/Projects/${TARGET_FILE}/src")
    endif()

    if(WIN32)
        install_static_lib("${TARGET_FILE}")
        # correct cmake, who puts the DLLs into the lib folder even though the exe needs it at runtime
        post_build_copy(
            "${TARGET_FILE}"
            "${PF}${TARGET_FILE}.${SH}"
            "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/lib"
            "${DLL_DIR}/../bin"
        )
    endif()

    if(${release})
        set(lib_destination "../Libraries") # only release global dlls go to libraries
    else()
        set(lib_destination "Applications")
        # debug and release EXEs go to Applications 
        # releae gets DLLs from the libraries folder
        # for debug EXEs to work, they need to be in the same path as the app
        # so they must go with the exe to applications
    endif()
    # copy the DLL/SO file
    post_build_copy(
        "${TARGET_FILE}"
        "${PF}${TARGET_FILE}.${SH}"
        "${DLL_DIR}"
        "${INSTALL_PREFIX}/${lib_destination}/Build/${BUILD_TYPE}/bin"
    )
    
    post_build_copy(
        "${TARGET_FILE}"
        "${TARGET_FILE}.${OBJ}"
        "${OBJ_DIR}"
        "${INSTALL_PREFIX}/${lib_destination}/Build/${BUILD_TYPE}/lib"
    )

endfunction()

# DEBUG MESSAGES
# MESSAGE("==================================================================")
# MESSAGE("FULL_PATH        = ${FULL_PATH}")
# MESSAGE("SOURCE_PATH      = ${SOURCE_PATH}")
# MESSAGE("FILE_NAME        = ${FILE_NAME}")
# MESSAGE("BASE             = ${BASE}")

function(INSTALL_VISUAL_STUDIO_SOLUTION)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        # FULL_PATH is the full path of the file's install location (not the solution/project location)
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${FULL_PATH}" FULL_LEN)
        string(LENGTH   "${CMAKE_SOURCE_DIR}/Solution/" SLN_FOLDER_LEN)
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
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(INSTALL_VISUAL_STUDIO_SOLUTION)

function(INSTALL_VISUAL_STUDIO_PROJECT)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        # FULL_PATH is the full path of the file's install location (not the solution/project location)
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${FULL_PATH}" FULL_LEN)
        string(LENGTH   "${CMAKE_SOURCE_DIR}/Project/" SLN_FOLDER_LEN)
        string(LENGTH   "${FILE_NAME}" FILE_LEN)

        string(LENGTH ${BUILD_DIR}   build_dir_len)
        string(LENGTH ${INSTALL_DIR} install_dir_len)


        math(EXPR RELATIVE_LEN   "${FULL_LEN} - ${build_dir_len} - ${install_dir_len}")

        math(EXPR RELATIVE_LEN   "${FULL_LEN} - ${SLN_FOLDER_LEN} - ${FILE_LEN}") # get the relative folder path by removing the sln path & file path
        string(SUBSTRING  ${FULL_PATH} ${SLN_FOLDER_LEN} ${RELATIVE_LEN} RELATIVE_PATH) # take only what is from the sln path + the relative path distnace
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}") # fix slashes on relative path

        # message("FULL_PATH     = ${FULL_PATH}")
        # message("Solution Path = ${CMAKE_SOURCE_DIR}/Project/")
        # message("RELATIVE_PATH = ${RELATIVE_PATH}")

        # message("BUILD_DIR     = ${BUILD_DIR}")
        # message("INSTALL_DIR   = ${INSTALL_DIR}")
        
        # message("--")

        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(INSTALL_VISUAL_STUDIO_PROJECT)

function(CONFIGURE_VISUAL_STUDIO_PROJECT)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        # FULL_PATH is the full path of the file's install location (not the solution/project location)
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${FULL_PATH}" FULL_LEN)
        string(LENGTH   "${RADICAL_BASE}/Libraries/Projects/${LIB}" PROJ_FOLDER_LEN)
        string(LENGTH   "${FILE_NAME}" FILE_LEN)

        math(EXPR RELATIVE_LEN   "${FULL_LEN}  - ${PROJ_FOLDER_LEN} - ${FILE_LEN}")
        string(SUBSTRING  ${FULL_PATH} ${PROJ_FOLDER_LEN} ${RELATIVE_LEN} RELATIVE_PATH)
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}")

        # message("FULL_PATH     = ${FULL_PATH}")
        # message("Project Path  = ${RADICAL_BASE}/Libraries/Projects/${LIB}")
        # message("RELATIVE_PATH = ${RELATIVE_PATH}")
        # message("--")

        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(CONFIGURE_VISUAL_STUDIO_PROJECT)

macro(install_symlink filepath sympath) # Windows requires admin right to create sym links
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
endmacro(install_symlink)

