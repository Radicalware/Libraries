

set(VLD_DIR "C:/Program Files (x86)/Visual Leak Detector")
set(VLD_LIB "${VLD_DIR}/lib/Win64")
set(VLD_TARGET "${VLD_LIB}/vld.lib")
include_directories("${VLD_DIR}/include")
find_library("${VLD_LIB}" NAMES vld)
