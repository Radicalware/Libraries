include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalQt6Solution InPrivateLibs InPublicLibs)

    add_definitions(-DQtAPP)

    set(SOLUTION "${CMAKE_SOURCE_DIR}/Solution")

    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    find_package(re2)

    find_package(
        Qt6  REQUIRED COMPONENTS 
            Core
            GUI
            Widgets
            Qml 
            Quick 
            QuickControls2
            QuickTemplates2
    )

    #find_package(PThreads4W REQUIRED)

    add_definitions(
        ${Qt6Widgets_DEFINITIONS} 
        ${QtQml_DEFINITIONS} 
        ${Qt6Quick_DEFINITIONS}
        ${Qt6Network_DEFINITIONS}
    )

    #find_path(VULKAN_HEADERS_INCLUDE_DIRS1 "${VCPKG_INCLUDE}/vk_video")
    #ind_path(VULKAN_HEADERS_INCLUDE_DIRS2  "${VCPKG_INCLUDE}/vulkan")
    #set(VULKAN_HEADERS_INCLUDE_DIRS "${VULKAN_HEADERS_INCLUDE_DIRS1};${VULKAN_HEADERS_INCLUDE_DIRS2}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt6Widgets_EXECUTABLE_COMPILE_FLAGS}")

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    list(APPEND QML_DIRS "${SOLUTION}/view")
    list(APPEND QML_DIRS "${SOLUTION}/view/Backend")
    list(APPEND QML_DIRS "${SOLUTION}/view/Constants")
    list(APPEND QML_DIRS "${SOLUTION}/view/Mods")
    list(APPEND QML_DIRS "${SOLUTION}/view/Support")
    set(QML_IMPORT_PATH  "${QML_DIRS}" CACHE STRING "Qt Creator extra qml import paths" FORCE)

    qt6_add_resources(QT_RESOURCES "${SOLUTION}/files.qrc")

    SetLocalInstallDirs()
    FindProgramFiles(SolutionFiles "${CMAKE_CURRENT_SOURCE_DIR}/Solution")

    add_executable(${THIS}
        ${SolutionFiles}
        ${QT_RESOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/Solution/resource/exe/pic.rc
    )

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
        "${Vulkan_INCLUDE_DIR}"
    )

    if(${debug} OR ${BuildAll})        
        foreach(Lib IN LISTS ${InPrivateLibs})
            message(" >> Linking Static Radical::${Lib}")
            target_link_libraries(${THIS} "Radical::${Lib}")
        endforeach()
    endif()

    PrintList(TargetLibs)
    target_link_libraries(${THIS}

        ${TargetLibs}

        Qt6::Core
        Qt6::Widgets
        Qt6::Qml
        Qt6::Quick
        Qt6::QuickControls2
        Qt6::QuickTemplates2

        #PThreads4W::PThreads4W
    )

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