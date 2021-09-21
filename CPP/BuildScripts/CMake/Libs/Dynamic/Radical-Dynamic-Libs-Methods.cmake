﻿macro(PrintEXEsAndDLLs)
    foreach(SharedLib ${SharedLibs})
        if(${FirstLibLoop})
            message("DLL >> ${SharedLib}")
        endif()
        LinkDynamic(${EXE} ${SharedLib})
    endforeach()

    if(${FirstLibLoop})
        message("------------------------------------")
    endif()
    set(FirstLibLoop OFF)
    message("EXE >> ${EXE}")    
endmacro()


function(LinkAllSharedLibs Target)
    foreach(DLL ${SharedLibs})
        message("DLL >> ${DLL}")
        LinkDynamic(${Target} ${DLL})
    endforeach()
endfunction()

# for this to run well in Visual Studio do the following...
# Tools > Options > Projects and Solutions > Build and Run > 
#    uncheck "Only build startup project and dependinces on run"
function(LinkDynamic TARGET_FILE DYNAMIC_LIB)
    if(${IsApp} AND (${debug} OR ${BuildAll}))
        if(IsWindows)
            set(SharedLibLinker "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/lib/${PF}${DYNAMIC_LIB}.${ST}")
            target_link_libraries(${TARGET_FILE}  ${SharedLibLinker})
        else()
            target_link_libraries(${TARGET_FILE}  "${CMAKE_SOURCE_DIR}/Build/${OS_TYPE}/${BUILD_TYPE}/${BUILD_TYPE}/bin/${PF}${DYNAMIC_LIB}.${SH}")
        endif()
    else()
        if(IsWindows) # windows injects the .lib into the .exe to locate the .dll file at runtime
            target_link_libraries(${TARGET_FILE} ${EXT_BIN_PATH}/lib/${PF}${DYNAMIC_LIB}.${ST})
        else()
            target_link_libraries(${TARGET_FILE} ${EXT_BIN_PATH}/bin/${PF}${DYNAMIC_LIB}.${SH})
        endif()
    endif()
endfunction()
