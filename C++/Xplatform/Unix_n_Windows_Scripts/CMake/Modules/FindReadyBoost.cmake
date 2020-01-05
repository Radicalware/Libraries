

if(WIN32)
    SET(BOOST_ROOT        "D:/Application_Engines/Boost")
    SET(BOOST_INCLUDEDIR  "D:/Application_Engines/Boost/boost")
    SET(BOOST_LIBRARYDIR  "D:/Application_Engines/Boost/libs")

    set(Boost_INCLUDE_DIR "D:/Application_Engines/Boost")
    set(CMAKE_MODULE_PATH  "${CMAKE_MODULE_PATH}:C:/Program Files/CMake/share/cmake-3.16/Modules")
    set(CMAKE_MODULE_PATH  "${CMAKE_MODULE_PATH}:D:/Application_Engines/Boost/tools/boost_install")
    set(CMAKE_MODULE_PATH  "${CMAKE_MODULE_PATH}:D:/Application_Engines/Boost")

endif()

SET(Boost_NO_BOOST_CMAKE ON)
set(Boost_USE_STATIC_LIBS      ON)  # only find static libs
set(Boost_USE_MULTITHREADED    ON)
set(Boost_USE_STATIC_RUNTIME   OFF) 

set(Boost_USE_DEBUG_LIBS       OFF) # ignore debug libs and 
set(Boost_USE_RELEASE_LIBS     ON)  # only find release libs 

message(">>> Boost Configured <<<")

# Example Usage
# find_package( Boost 1.71.0 EXACT REQUIRED COMPONENTS regex )


