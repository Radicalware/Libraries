cmake_minimum_required(VERSION 3.25)

FindStaticLib("Stats")
if(UsingNVCC)
    ConfigCUDA("Stats")
else()
    set_target_properties(Stats PROPERTIES LINKER_LANGUAGE CXX)
endif()

