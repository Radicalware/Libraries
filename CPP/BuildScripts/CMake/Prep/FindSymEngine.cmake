# Build source with MD & MDd
# Properties >> C++ >> Code Generation

# Also, it may require msys2 
# which requires disabling ASLR for those MSYS2 exes

set(ThisLibName "symengine")
set(SymEngine_Root "D:/Libs/${ThisLibName}")

find_path(SymEngine_INCLUDE_DIR
  NAMES symengine/symbol.h
  PATHS "${SymEngine_Root}" "${SymEngine_Root}/${ThisLibName}"
)
find_path(SymEngineBuild_INCLUDE_DIR
  NAMES symengine/symengine_config.h
  PATHS "${SymEngine_Root}/build" "${SymEngine_Root}/build/${ThisLibName}"
)

find_library(SymEngine_LIBRARY_RELEASE
  NAMES symengine
  PATHS "${SymEngine_Root}/build/${ThisLibName}/Release"
)

find_library(SymEngine_LIBRARY_DEBUG
  NAMES symengine
  PATHS "${SymEngine_Root}/build/${ThisLibName}/Debug"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SymEngine DEFAULT_MSG
    SymEngine_INCLUDE_DIR 
    SymEngine_LIBRARY_RELEASE 
    SymEngine_LIBRARY_DEBUG
)

if(SymEngine_FOUND)
  set(SymEngine_INCLUDE_DIRS ${SymEngine_INCLUDE_DIR})
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(SymEngine_LIBRARIES ${SymEngine_LIBRARY_DEBUG})
    mark_as_advanced(SymEngine_INCLUDE_DIR SymEngine_LIBRARY_DEBUG)
  else()
    set(SymEngine_LIBRARIES ${SymEngine_LIBRARY_RELEASE})
    mark_as_advanced(SymEngine_INCLUDE_DIR SymEngine_LIBRARY_RELEASE)
  endif()
endif()

include_directories(${SymEngine_INCLUDE_DIRS})
include_directories(${SymEngineBuild_INCLUDE_DIR})

set(GMP_DLL "gmp-10.dll")
MakeBinaryFileCopy(
    "${SymEngine_Root}/build/benchmarks/Release/${GMP_DLL}"
    "${OUTPUT_DIR}/bin/${GMP_DLL}")


list(APPEND TargetLibs ${SymEngine_LIBRARIES})
