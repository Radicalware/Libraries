cmake_minimum_required(VERSION 3.10)

SET(THIS ${ArgProjectName})
set(BuildAll ON)

# -------------------------- PRE-CONFIG -------------------------------------------------

if(WIN32)
    set(RADICAL_PATH "C:/Source/CMake/Modules")
else()
    set(RADICAL_PATH "/opt/Radicalware/CMake_Modules")
endif()

include("${RADICAL_PATH}/Radical-App-Config.cmake")


# cp `
#   "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\extras\visual_studio_integration\MSBuildExtensions"
#   "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations"

# Env Vars
    # CUDA_PATH             = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
    # CUDA_PATH_V11_3       = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
    # CMAKE_CUDA_COMPILER   = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin\nvcc.exe
    # CUDAToolkit_ROOT      = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin

    # VCPKG_DEFAULT_TRIPLET = x64-windows

# CUDA C++ COMPILER OPTIONS: -std=c++20
# CUDA C++ LINKER   OPTIONS: check "inherit from parent"

# GET CUDA FROM NVIDIA
# GET QT FROM QT.IO
# DO NOT GET CUDA OR QT FROM VCPKG
# CUDA 11 & QT 6


# -------------------------- BUILD ------------------------------------------------------

project(${THIS})
#project(${THIS} LANGUAGES CXX CUDA) # TODO: RE-ADD
#find_package(PrepareCuda) # TODO: RE-ADD

SetLocalInstallDirs()

set(EXES "")
set(AllFiles "")

macro(AddExecutable ExePathPart ExeName)
    set(ExePath "${CMAKE_SOURCE_DIR}/../../${ExePathPart}/${ExeName}/Examples/Solution")
    unset(EXE_SolutionFiles)
    set(EXE_SolutionFiles "")
    file(GLOB EXE_SolutionFiles
           ${ExePath}/include/*.h
           ${ExePath}/include/*.cuh
           ${ExePath}/src/*.cpp
           ${ExePath}/src/*.cu
           )
    
    include_directories(${THIS} PRIVATE "${ExePath}/include")
    add_executable(  EX_${ExeName} ${EXE_SolutionFiles})
    list(APPEND      AllFiles      ${EXE_SolutionFiles})
    list(APPEND      EXES          EX_${ExeName})
endmacro(AddExecutable)

# !!!! ADD LIBS TO ALL 3 LOCATIONS !!!!

# (This is location 1 of 3)
AddExecutable("FunctionalSTL" "AC")
AddExecutable("FunctionalSTL" "RE")
AddExecutable("FunctionalSTL" "MC")

AddExecutable("ExtendedSTL" "Memory")
AddExecutable("ExtendedSTL" "xvector")
AddExecutable("ExtendedSTL" "xstring")
AddExecutable("ExtendedSTL" "xmap")

AddExecutable("GeneralPurposeLibs" "Macros")
AddExecutable("GeneralPurposeLibs" "Nexus")
AddExecutable("GeneralPurposeLibs" "Timer")
AddExecutable("GeneralPurposeLibs" "Date")
AddExecutable("GeneralPurposeLibs" "OS")
AddExecutable("GeneralPurposeLibs" "SYS")
AddExecutable("GeneralPurposeLibs" "Socket")
AddExecutable("GeneralPurposeLibs" "Stash")
AddExecutable("GeneralPurposeLibs" "AES")

AddExecutable("ModdedObjects" "CC")

include_directories(${THIS} PRIVATE
        ${InstalledIncludeDirs}
        ${AllFiles}
)

# (This is location 2 of 3)

find_package(VLD)
find_package(re2)

find_package(RE)
find_package(AC)
find_package(MC)

find_package(Iterator)
find_package(Memory)
find_package(Nexus)
find_package(xvector)
find_package(xstring)
find_package(Macros)
find_package(xmap)

find_package(OS)
find_package(SYS)
find_package(Timer)
find_package(Date)
find_package(cc)
find_package(JSON)
find_package(Stash)
find_package(AES)
find_package(Socket)
# find_package(CudaBridge) # TODO: RE-ADD

find_package(cpprestsdk CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

message("====================================")
PrintLibs()
message("------------------------------------")
set(FirstLoop ON)
foreach(EXE ${EXES})
    target_link_libraries(${EXE} PRIVATE

            # (This is location 3 of 3)
            
            Radical::AC
            Radical::MC

            Radical::Iterator
            Radical::Memory
            Radical::Nexus
            Radical::xvector
            Radical::xstring
            Radical::Macros
            Radical::xmap
            Radical::JSON
            Radical::Stash
            Radical::Socket
            # Radical::CudaBridge # TODO: RE-ADD
            # -----------------------------------------
            
            cpprestsdk::cpprest
            cpprestsdk::cpprestsdk_zlib_internal
            cpprestsdk::cpprestsdk_brotli_internal

            mongo::bsoncxx_shared
            mongo::bson_shared
            mongo::mongocxx_shared

            nlohmann_json
            nlohmann_json::nlohmann_json

            OpenSSL::SSL
            OpenSSL::Crypto
        )

        PrintEXEsAndDLLs()

        set_target_properties(${EXE} PROPERTIES COMPILE_DEFINITIONS DLL_EXPORT=1)
        if(${debug})
            LinkStatic(${EXE} vld)
        endif()
        LinkStatic(${EXE} re2)

        LinkAllSharedLibs(${EXE})
        SetAllDependenciesOn(${EXE})
        LinkAllStaticLibs(${EXE})
        target_link_libraries(${EXE} PRIVATE "${UsedVcpkgLibs}")
endforeach()
message("====================================")

# -------------------------- POST-CONFIG ------------------------------------------------

set(TargetProject "EX_xstring")
SetVisualStudioFilters("Solution" "${AllFiles}")
include("${RADICAL_PATH}/Radical-App-Install.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------- END --------------------------------------------------------
