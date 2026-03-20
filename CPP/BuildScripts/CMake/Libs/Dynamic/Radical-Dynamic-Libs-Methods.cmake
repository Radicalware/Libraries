macro(PrintEXEsAndDLLs)
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
    target_link_libraries(${TARGET_FILE} PRIVATE ${DYNAMIC_LIB})
endfunction()
