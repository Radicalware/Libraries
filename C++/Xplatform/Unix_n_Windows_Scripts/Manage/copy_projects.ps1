#!/usr/bin/env pwsh

Write-Host
Write-Host "----------------------------------------------------------------------"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!


$General_Purpose_Libs = 'General_Purpose_Libs'
$eXtended_STL         = 'eXtended_STL'
$Functional_STL       = 'Functional_STL'
$Modded_Objects       = 'Modded_Objects'

Write-Host "Copying Project Folders: "
Write-Host "1. $General_Purpose_Libs : Nexus, Socket, OS, 
                          SYS, Iterator, Timer, Date, CudaBridge"
Write-Host "2. $eXtended_STL         : xvector, xstring, xmap"
Write-Host "3. $Functional_STL       : re, ac, mc"
Write-Host "4. $Modded_Objects       : cc, re2"

# ALERT!!! AFTER MAKING MODS, BE SURE TO COPY THIS FILE TO YOUR POWRSHELL PATH !!!

class Lib_Handler
{
    [string] $dir;
    
    [string] $build_lib_dir     = 'C:\Source\Radicalware\git\Libraries\C++\Xplatform'
    [string] $build_module_dir  = 'C:\Source\Radicalware\git\Libraries\C++\Modules'

    [string] $cmake_install_dir = 'C:\Source\CMake\Radicalware\Libraries\Projects'
    [string] $cmake_header_dir  = 'C:\Source\CMake\Radicalware\Libraries\Headers'

    [string] $inc = "include\*"

    [void] CopyLib([string] $lib)
    {    
        Copy-Item  "$($this.cmake_install_dir)\$lib\*"  "$($this.build_lib_dir)\$($this.dir)\$lib\Project" -Recurse -Force
        
        # It is better to use soft links
        # Copy-Item  "$($this.cmake_install_dir)\$lib\$($this.inc)"  "$($this.cmake_header_dir)" -Recurse -Force

    }

    [void] CopyModule([string] $lib)
    {    
        Copy-Item  "$($this.cmake_install_dir)\$lib\*"  "$($this.build_module_dir)\$($this.dir)\$lib\Project" -Recurse -Force
        
        # It is better to use soft links
        # Copy-Item  "$($this.cmake_install_dir)\$lib\$($this.inc)"  "$($this.cmake_header_dir)" -Recurse -Force

    }
}

$handler = [Lib_Handler]::new()

$handler.dir = $General_Purpose_Libs
$handler.CopyLib("Nexus")
$handler.CopyLib("Socket")
$handler.CopyLib("OS")
$handler.CopyLib("SYS")
$handler.CopyLib("Iterator")
$handler.CopyLib("Timer")
$handler.CopyLib("Date")
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
