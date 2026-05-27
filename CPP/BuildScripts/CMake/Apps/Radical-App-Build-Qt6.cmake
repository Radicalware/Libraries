include("${RADICAL_PATH}/Radical-App-Config.cmake")

macro(BuildRadicalQt6Solution InPrivateLibs InPublicLibs)

    RunSilentPowershell("rm ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format_11")
    RunSilentPowershell("New-Item ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format_11 | Out-Null")

    add_definitions("-DQtAPP")

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
        "-D${Qt6Widgets_DEFINITIONS}"
        "-D${QtQml_DEFINITIONS}" 
        "-D${${Qt6Quick_DEFINITIONS}}"
        "-D${Qt6Network_DEFINITIONS}"
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

    qt_add_executable(${THIS}
        "${SolutionFiles}"
        "${QT_RESOURCES}"
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/resource/exe/pic.rc"
        "${QmlFiles}"
        "${AssetsFiles}"
    )

    # Deploy resources to build folder/package directly
    # comment for publishing
    # deploy_resources("${QmlFiles};${AssetsFiles}")

    # Add QML files and resources to QML module to included them via QRC automatically:
    qt_add_qml_module(${THIS}
        URI com.github.Radicalware.${THIS}
        VERSION 1.0
        QML_FILES ${QmlFiles}
        RESOURCES ${AssetsFiles}
        NO_RESOURCE_TARGET_PATH
    )

    if(BxDebug)
        target_compile_definitions(${THIS} PRIVATE
            QT_QML_DEBUG
            QtAPP
            ABSL_CONSUME_DLL
            CMAKE_INTDIR="Debug"
        )
    endif()

    if (MSVC) 
        target_compile_options(${THIS} PRIVATE /std:c++latest) 
    endif()

    target_include_directories(${THIS} PRIVATE
        ${InstalledIncludeDirs}
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/Solution/controller/include"
        "${Vulkan_INCLUDE_DIR}"
    )

    foreach(LoLib IN LISTS ${InPrivateLibs})
        find_package("${LoLib}")
    endforeach()

    foreach(LoLib IN LISTS ${InPublicLibs})
        find_package("${LoLib}")
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

    # See https://github.com/Radicalware/Libraries/pull/2
    # I was requested to put these here, as opposed to updating the LinkStatic & LinkDynamic functions

    # LinkAllSharedLibs(${THIS})
    # LinkStatic(${THIS} re2)
    # LinkAllStaticLibs(${THIS})
    foreach(DLL ${SharedLibs})
        message("DLL >> ${DLL}")
        target_link_libraries(${THIS} PRIVATE ${DLL})
    endforeach()
    SetAllDependenciesOn(${THIS})
    foreach(LIB ${StaticLibs} re2)
        message("LIB >> ${LIB}")
        target_link_libraries(${THIS} PRIVATE ${LIB})
    endforeach()

    set(TargetProject ${THIS})
    SetVisualStudioFilters("Solution" "${SolutionFiles}")
    if(${debug} OR ${BuildAll})
        include("${RADICAL_PATH}/Radical-App-Install.cmake")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endmacro()
