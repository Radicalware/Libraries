
if($args[0] -eq "gl") # Get Libs
{
    $($(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects).`
        foreach({";$($_.FullName)\Include" }) | 
        Out-String) -replace "`r`n", ""
    return;
}

$ExeName = "Test"


$UsingNVCC = [regex]::new("^.*(cuh|cu)$").match($args[0]).Success;
# TODO: >>>> add a DLL build later

# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include" 
# -cudart 'none','shared','static' (Used to specify a Lib Type)

if($UsingNVCC -eq $true)
{
    $LibsList = $($(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Build\Release\lib).foreach({" --library $($_.FullName )" }).replace(".obj","").replace(".lib","").split("`n") | Get-Unique | Out-String).Replace("`r`n"," ");
    $IncludeDirs =  $($(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects).foreach({" -I`"$($_.FullName)\Include`"" }) | Out-String).Replace("`r`n"," ");


    Write-Host "Building CUDA"
    Invoke-Expression "nvcc.exe -gencode `"arch=compute_61,code=sm_61`" --use-local-env
        -ccbin `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\HostX64\x64`"
        -x cu -I`"D:\AIE\vcpkg\installed\x64-windows\include`"
        --library-path  C:\Source\CMake\Radicalware\Libraries\Build\Release\lib
        $LibsList $IncludeDirs
        --machine 64 -cudart static -std=c++20
        -g `-D_WINDOWS -DReleaseOn -D`"CMAKE_INTDIR=Release`"  -DReleaseOn -DUsingNVCC -D`"CMAKE_INTDIR=Release`"
        -Xcompiler `"/EHsc, /W1, /nologo, /O2, /FS,  /MD, /GR, /std:c++20 `"
        -Xlinker `"/NODEFAULTLIB:library, /IGNORE:4098`"
         $args -o Test.exe".Replace("`r`n"," ")
    
    del "$ExeName.exp"
    del "$ExeName.lib"
    del "$ExeName.pdb"
    del "vc140.pdb"
}
else
{
    Write-Host "Building Standard"
    cl.exe $args /Fe:Test /EHsc /O2 /nologo /std:c++20 `
        $(Get-ChildItem C:\Source\CMake\Radicalware\Libraries\Projects).`
            foreach({"-I`"$($_.FullName)\Include`"" })
    del $args[0].replace("`.cpp",".obj") | Out-Null
}

write-host "--------------------------------- EXECUTING -------------------------------------------"

./Test.exe

write-host "------------------------------------ DONE ---------------------------------------------"

#del Test.exe


