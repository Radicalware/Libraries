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
    )

    if(${debug} OR ${BuildAll})        
        foreach(Lib IN LISTS ${InPrivateLibs})
            target_link_libraries(${THIS} "Radical::${Lib}")
        endforeach()
    endif()

    LinkAllSharedLibs(${THIS})
    SetAllDependenciesOn(${THIS})

    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
