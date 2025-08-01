

set(VLD_DIR "C:/Program Files (x86)/Visual Leak Detector")

include_directories("${VLD_DIR}/include")
set(VLD_LIB         "${VLD_DIR}/lib/Win64")
set(VLD_TARGET      "${VLD_DIR}/lib/Win64/vld.lib")

find_library("${VLD_LIB}" NAMES vld)

if(${BuildAll})
    MakeBinaryFileCopy(${VLD_TARGET} "${OUTPUT_DIR}/lib/vld.lib")
else()
    MakeBinaryFileCopy(${VLD_TARGET} "${EXT_BIN_PATH}/vld.lib")
endif()

