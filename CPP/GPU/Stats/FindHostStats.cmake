cmake_minimum_required(VERSION 3.25)

FindStaticLib("HostStats")
if(UsingNVCC)
    message(FATAL_ERROR "Use Stats instead of Host Stats if compiling with NVCC")
endif()

