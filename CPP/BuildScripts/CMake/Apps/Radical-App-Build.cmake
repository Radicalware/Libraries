include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalSolution InPrivateLibs InPublicLibs)
    
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

    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
