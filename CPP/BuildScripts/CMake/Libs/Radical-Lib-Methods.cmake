function(SetStaticDependenciesOn Target)
    if(${BuildAll})
        foreach(Lib ${StaticLibs})
            add_dependencies(${Target} ${Lib})
        endforeach()
    endif()
endfunction()

