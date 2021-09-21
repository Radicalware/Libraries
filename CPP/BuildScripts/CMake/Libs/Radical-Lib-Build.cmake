﻿include("${RADICAL_PATH}/Radical-Lib-Config.cmake")

macro(BuildRadicalProject InLibType InPrivateLibs InPublicLibs)
    # -------------------------- CONFIG -----------------------------------------------------
    # InLibType = {MODLULE : DLL},{STATIC : LIB}    

    FindProgramFiles(ProjectFiles "${BUILD_DIR}")
    add_library(${THIS} ${InLibType} ${ProjectFiles})
    add_library(Radical::${THIS} ALIAS ${THIS})

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

    # -------------------------- INSTALL ----------------------------------------------------

    include("${RADICAL_PATH}/Radical-Lib-Install.cmake")
    SetVisualStudioFilters("Project" "${ProjectFiles}")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # -------------------------- END --------------------------------------------------------
endmacro()
