function(SetStaticDependenciesOn Target)
    if(${BuildAll})
        foreach(LoLIB ${StaticLibs})
            add_dependencies(${Target} ${Lib})
        endforeach()
    endif()
endfunction()

