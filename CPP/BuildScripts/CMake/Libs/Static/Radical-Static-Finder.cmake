macro(FindStaticLib LIB)
    # -------------------------- PRE-CONFIG ---------------------------------------
    list(APPEND InstalledIncludeDirs   "${PROJECT_DIR}/${LIB}/include")
    list(APPEND StaticLibs ${LIB})
    if(${release} AND NOT ${BuildAll})
        LinkStatic(${THIS} ${LIB})
        return()
    endif()
    # -------------------------- BUILD --------------------------------------------

    UNSET(ProjectFiles)
    FindProgramFiles(ProjectFiles "${PROJECT_DIR}/${LIB}")
    add_library(${LIB} STATIC ${ProjectFiles})
    add_library(Radical::${LIB} ALIAS ${LIB})
    target_include_directories(${LIB} PUBLIC ${InstalledIncludeDirs})
    target_link_libraries(${LIB} ${PreStaticLibLst})
    list(APPEND PreStaticLibLst "${LIB}")

    # -------------------------- POST-CONFIG --------------------------------------
    SetVisualStudioFilters("Projects.${LIB}" "${ProjectFiles}")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    # -------------------------- END ----------------------------------------------
endmacro()
