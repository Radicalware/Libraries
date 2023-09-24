include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalSolution InPrivateLibs InPublicLibs)
    
    add_compile_definitions("BuildingApp")
    
    find_package(re2)
    find_package(VLD)
    
    SetLocalInstallDirs()
    FindProgramFiles(SolutionFiles "${CMAKE_CURRENT_SOURCE_DIR}/Solution")
    add_executable(${THIS} ${SolutionFiles})

    foreach(Lib IN LISTS ${InPrivateLibs})
        find_package("${Lib}")
    endforeach()

    foreach(Lib IN LISTS ${InPublicLibs})
        find_package("${Lib}")
    endforeach()

    target_include_directories(${THIS} PRIVATE
        ${InstalledIncludeDirs}
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/controller/include"
    )

    if(${debug} OR ${BuildAll})        
        foreach(Lib IN LISTS ${InPrivateLibs})
            message(" >> Linking Static Radical::${Lib}")
            target_link_libraries(${THIS} "Radical::${Lib}")
        endforeach()
    endif()

    if(${debug})
        message("(Debug Build)")
        # target_link_libraries(${THIS} vld.lib) # unreliable
        message("Linking: ${VLD_TARGET}")
        target_link_libraries(${THIS} "${VLD_TARGET}")
    else()
        message("(Release Build)")
    endif()

    PrintList(TargetLibs)
    target_link_libraries(${THIS} ${TargetLibs})

    # if(${debug} OR ${BuildAll})        
    #     foreach(Lib IN LISTS ${InPublicLibs})
    #         Fails because you can only link... INTERFACE, OBJECT, STATIC or SHARED
    #         This does NOT include Module types. DLLs are Module types
    #         message(" >> Linking Dynamic Radical::${Lib}")
    #         target_link_libraries(${THIS} "Radical::${Lib}")
    #     endforeach()
    # endif()

    LinkAllSharedLibs(${THIS})
    SetAllDependenciesOn(${THIS})
    LinkStatic(${THIS} re2)

    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
