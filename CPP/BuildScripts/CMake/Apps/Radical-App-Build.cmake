include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalSolution InPrivateLibs InPublicLibs)
    
    add_compile_definitions("BuildingApp")
    
    find_package(re2)
    find_package(VLD)
    
    SetLocalInstallDirs()
    FindProgramFiles(SolutionFiles "${CMAKE_CURRENT_SOURCE_DIR}/Solution")
    add_executable(${THIS} ${SolutionFiles})

    if (MSVC) 
        target_compile_options(${THIS} PRIVATE /std:c++latest) 
        #target_link_options(${THIS} PRIVATE /NODEFAULTLIB:MSVCRT)
    endif()

    # ----------------------------------------------------------------------------
    # Get the FindX.cmake for each array input
    foreach(Lib IN LISTS ${InPrivateLibs})
        find_package("${Lib}")
    endforeach()

    foreach(Lib IN LISTS ${InPublicLibs})
        find_package("${Lib}")
    endforeach()
    # ----------------------------------------------------------------------------
    # Set Include directories
    target_include_directories(${THIS} PRIVATE
        ${InstalledIncludeDirs}
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/controller/include"
    )
    # ----------------------------------------------------------------------------
    # Link the Radicalware Targets from your FindX.Cmake list

    if(${debug})
        message("(Debug Build)")
        # target_link_libraries(${THIS} vld.lib) # unreliable
        message("Linking: ${VLD_TARGET}")
        target_link_libraries(${THIS} PRIVATE "${VLD_TARGET}")
    else()
        message("(Release Build)")
    endif()

    LinkAllSharedLibs(${THIS})
    SetAllDependenciesOn(${THIS})
    LinkStatic(${THIS} re2)
    LinkAllStaticLibs(${THIS})
    target_link_libraries(${THIS} PRIVATE "${UsedVcpkgLibs}")

    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
