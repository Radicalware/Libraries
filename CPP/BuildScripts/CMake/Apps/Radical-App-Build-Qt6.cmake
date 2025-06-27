include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalQt6Solution InPrivateLibs InPublicLibs)

    RunSilentPowershell("rm ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format_11")
    RunSilentPowershell("New-Item ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format_11 | Out-Null")

    add_definitions(-DQtAPP)

    set(SOLUTION "${CMAKE_SOURCE_DIR}/Solution")

    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(QT_QML_GENERATE_QMLLS_INI ON)
    # set(QT_DEBUG_FIND_PACKAGE ON)

    if(debug)
        find_package(VLD)
    endif()

    find_package(re2)
    find_package(Threads)

    find_package(Qt6  REQUIRED COMPONENTS 
        Core
        GUI
        Widgets
        Qml 
        Quick 
        QuickControls2
        QuickTemplates2
        Graphs
        Charts
        DataVisualization
    )

    add_definitions(
        ${Qt6Widgets_DEFINITIONS} 
        ${QtQml_DEFINITIONS} 
        ${Qt6Quick_DEFINITIONS}
        ${Qt6Network_DEFINITIONS}
    )

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
        ${QmlFiles}
        ${AssetsFiles}
    )

    # Deploy resources to build folder/package directly
    # comment for publishing
    # deploy_resources("${QmlFiles};${AssetsFiles}")

    # Add QML files and resources to QML module to included them via QRC automatically:
    qt_add_qml_module(${THIS}
        URI BasicApp
        VERSION 1.0
        QML_FILES ${QmlFiles}
        RESOURCES ${AssetsFiles}
        NO_RESOURCE_TARGET_PATH
    )

    if (MSVC) 
        target_compile_options(${THIS} PRIVATE /std:c++latest) 
        #target_link_options(${THIS} PRIVATE /NODEFAULTLIB:MSVCRT)
    endif()

    target_include_directories(${THIS} PRIVATE
        ${InstalledIncludeDirs}
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/controller/include"
        "${Vulkan_INCLUDE_DIR}"
    )

    foreach(Lib IN LISTS ${InPrivateLibs})
        find_package("${Lib}")
    endforeach()

    foreach(Lib IN LISTS ${InPublicLibs})
        find_package("${Lib}")
    endforeach()

    if(${debug})
        message("Linking: ${VLD_TARGET}")
        target_link_libraries(${THIS} PRIVATE ${VLD_TARGET})

    endif()

    PrintList(TargetLibs)
    target_link_libraries(${THIS} PRIVATE

        ${TargetLibs}

        Qt6::Core
        Qt6::Widgets
        Qt6::Qml
        Qt6::Quick
        Qt6::QuickControls2
        Qt6::QuickTemplates2
        Qt6::Graphs
        Qt6::Charts
        Qt6::DataVisualization
    )

    LinkAllSharedLibs(${THIS})
    SetAllDependenciesOn(${THIS})
    LinkStatic(${THIS} re2)
    LinkAllStaticLibs(${THIS})


    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
