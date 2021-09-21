macro(FindDynamicLib LIB)
    # -------------------------- PRE-CONFIG ---------------------------------------
    list(APPEND InstalledIncludeDirs   "${PROJECT_DIR}/${LIB}/include")
    list(APPEND SharedLibs "${LIB}")
    if(${release} AND NOT ${BuildAll})
        return()
    endif()
    # -------------------------- BUILD --------------------------------------------

    UNSET(ProjectFiles)
    FindProgramFiles(ProjectFiles "${PROJECT_DIR}/${LIB}")
    add_library(${LIB} MODULE ${ProjectFiles})
    add_library(Radical::${LIB} ALIAS ${LIB})
    target_include_directories(${LIB} PUBLIC ${InstalledIncludeDirs})
    target_link_libraries(${LIB} ${StaticLibs})
    SetStaticDependenciesOn(${LIB})
    set_target_properties(${LIB} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)

    # -------------------------- POST-CONFIG --------------------------------------
    SetVisualStudioFilters("Projects.${LIB}" "${ProjectFiles}")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    # -------------------------- END ----------------------------------------------
endmacro()
