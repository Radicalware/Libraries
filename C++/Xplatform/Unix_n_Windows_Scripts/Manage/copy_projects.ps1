#!/usr/bin/env pwsh

Write-Host
Write-Host "----------------------------------------------------------------------"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!


$General_Purpose_Libs = 'General_Purpose_Libs'
$eXtended_STL         = 'eXtended_STL'
$Functional_STL       = 'Functional_STL'
$Modded_Objects       = 'Modded_Objects'

Write-Host "Copying Project Folders: "
Write-Host "1. $General_Purpose_Libs : Macros, Nexus, Socket, OS, 
                          SYS, Iterator, Timer, Date
                          CudaBridge, Stash, JSON"
Write-Host "2. $eXtended_STL         : xvector, xstring, xmap"
Write-Host "3. $Functional_STL       : re, ac, mc"
Write-Host "4. $Modded_Objects       : cc, re2"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!

class Lib_Handler
{
    [string] $dir;
    
    [string] $build_lib_dir
    [string] $build_module_dir

    [string] $cmake_install_dir = 'C:\Source\CMake\Radicalware\Libraries\Projects'
    [string] $cmake_header_dir  = 'C:\Source\CMake\Radicalware\Libraries\Headers'

    [string] $inc = "include\*"

    Lib_Handler()
    {
        $this.build_lib_dir    = "$PSScriptRoot\..\.."
        $this.build_module_dir = "$PSScriptRoot\..\..\..\Modules"
    }

    [void] CopyLib([string] $lib)
    {    
        $From = "$($this.cmake_install_dir)\$lib\*"
        $To   = "$($this.build_lib_dir)\$($this.dir)\$lib\Project"
        # Write-Host "Copying: $From >> $To"

        try{
            if(Test-Path $From){
                Copy-Item -Recurse -Force $From $To
            }
        }catch
        {
            Write-Host "Can't Copy: $From >> $To"
        }
    }

    [void] CopyModule([string] $lib)
    {    
        $From = "$($this.cmake_install_dir)\$lib\*"
        $To   = "$($this.build_module_dir)\$($this.dir)\$lib\Project"
        # Write-Host "Copying: $From >> $To"

        try{
            if(Test-Path $From){
                Copy-Item -Recurse -Force $From $To
            }
        }catch
        {
            Write-Host "Can't Copy: $From >> $To"
        }
    }
}

$handler = [Lib_Handler]::new()

$handler.dir = $General_Purpose_Libs
$handler.CopyLib("Macros")
$handler.CopyLib("Nexus")
$handler.CopyLib("Socket")
$handler.CopyLib("OS")
$handler.CopyLib("SYS")
$handler.CopyLib("Iterator")
$handler.CopyLib("Timer")
$handler.CopyLib("Date")
$handler.CopyLib("JSON")
$handler.CopyLib("Stash")
$handler.CopyLib("CudaBridge")

$handler.dir = $eXtended_STL
$handler.CopyLib("xvector")
$handler.CopyLib("xstring")
$handler.CopyLib("xmap")

$handler.dir = $Functional_STL
$handler.CopyLib("re")
$handler.CopyLib("ac")
$handler.CopyLib("mc")

$handler.dir = $Modded_Objects
$handler.CopyLib("cc")
$handler.CopyLib("re2")


Write-Host "All Lib Folders Updated!!"
Write-Host "----------------------------------------------------------------------`n"
