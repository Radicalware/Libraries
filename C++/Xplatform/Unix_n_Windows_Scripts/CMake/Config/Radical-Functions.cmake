function(RENDER)
    foreach(VALUE IN ITEMS ${ARGN})
        message(">> ${VALUE}") # can't use VAR
    endforeach()
endfunction(RENDER)

MACRO(SUBDIRLIST RESULT CUR_DIR)
    FILE(GLOB CHILDREN RELATIVE ${CUR_DIR} ${CUR_DIR}/*)
    FOREACH(CHILD ${CHILDREN})
        IF(IS_DIRECTORY "${CUR_DIR}/${CHILD}")
            SUBDIRLIST(${RESULT} "${CUR_DIR}/${CHILD}")
        ELSE()
            LIST(APPEND ${RESULT} "${CUR_DIR}/${CHILD}")
        ENDIF()
    ENDFOREACH()
ENDMACRO()

        # DEBUG MESSAGES
        # MESSAGE("==================================================================")
        # MESSAGE("FULL_PATH        = ${FULL_PATH}")
        # MESSAGE("SOURCE_PATH      = ${SOURCE_PATH}")
        # MESSAGE("FILE_NAME        = ${FILE_NAME}")
        # MESSAGE("BASE             = ${BASE}")

function(CONFIGURE_VISUAL_STUDIO_SOLUTION)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${CMAKE_SOURCE_DIR}/Solution" BASE_LENG)
        string(LENGTH   "${FULL_PATH}" FULL_PATH_LENG)
        string(LENGTH   "${FILE_NAME}" FILE_LENG)
        math(EXPR FILTER_LENG "${FULL_PATH_LENG} - ${BASE_LENG} - ${FILE_LENG}" OUTPUT_FORMAT DECIMAL)

        string(SUBSTRING  "${FULL_PATH}" ${BASE_LENG} ${FILTER_LENG} RELATIVE_PATH)
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}")
        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(CONFIGURE_VISUAL_STUDIO_SOLUTION)

function(CONFIGURE_VISUAL_STUDIO_PROJECT)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        set(BASE   "${RADICAL_BASE}/Libraries/Projects/${LIB}")
        string(LENGTH "${BASE}" BASE_LENG)
        string(LENGTH "${FULL_PATH}" FULL_PATH_LENG)
        string(LENGTH "${FILE_NAME}" FILE_LENG)
        math(EXPR FILTER_LENG "${FULL_PATH_LENG} - ${BASE_LENG} - ${FILE_LENG}" OUTPUT_FORMAT DECIMAL)

        string(SUBSTRING  "${FULL_PATH}" ${BASE_LENG} ${FILTER_LENG} RELATIVE_PATH)
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}")
        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
endfunction(CONFIGURE_VISUAL_STUDIO_PROJECT)

function(CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT)
    foreach(FULL_PATH IN ITEMS ${ARGN})
        get_filename_component(SOURCE_PATH "${FULL_PATH}" PATH)
        get_filename_component(FILE_NAME   "${FULL_PATH}" NAME)

        string(LENGTH   "${CMAKE_SOURCE_DIR}/Project" BASE_LENG)
        string(LENGTH   "${FULL_PATH}" FULL_PATH_LENG)
        string(LENGTH   "${FILE_NAME}" FILE_LENG)
        math(EXPR FILTER_LENG "${FULL_PATH_LENG} - ${BASE_LENG} - ${FILE_LENG}" OUTPUT_FORMAT DECIMAL)

        string(SUBSTRING  "${FULL_PATH}" ${BASE_LENG} ${FILTER_LENG} RELATIVE_PATH)
        string(REPLACE "/" "\\\\" RELATIVE_PATH "${RELATIVE_PATH}")
        source_group("${RELATIVE_PATH}" FILES "${FULL_PATH}")
    endforeach()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${THIS})
endfunction(CONFIGURE_VISUAL_STUDIO_STANDALONE_PROJECT)

macro(install_symlink filepath sympath) # Windows requires admin right to create sym links
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${filepath} ${sympath})")
    install(CODE "message(\"-- Created symlink: ${sympath} -> ${filepath}\")")
endmacro(install_symlink)