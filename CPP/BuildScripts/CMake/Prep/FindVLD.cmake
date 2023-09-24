

set(VLD_DIR "C:/Program Files (x86)/Visual Leak Detector")

include_directories("${VLD_DIR}/include")
set(VLD_LIB         "${VLD_DIR}/lib/Win64")
set(VLD_TARGET      "${VLD_DIR}/lib/Win64/vld.lib")

find_library("${VLD_LIB}" NAMES vld)
