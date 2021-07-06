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
    
    [string] $build_dir         = 'C:\Source\Radicalware\Libraries'
    [string] $cmake_install_dir = 'C:\Source\CMake\Radicalware\Libraries\Projects'
    [string] $cmake_header_dir  = 'C:\Source\CMake\Radicalware\Libraries\Headers'

    [string] $inc = "include\*"

    [void] copy([string] $lib)
    {    
        Copy-Item  "$($this.cmake_install_dir)\$lib\*"  "$($this.build_dir)\$($this.dir)\$lib\Project" -Recurse -Force
        
        # It is better to use soft links
        # Copy-Item  "$($this.cmake_install_dir)\$lib\$($this.inc)"  "$($this.cmake_header_dir)" -Recurse -Force

    }
}

$handler = [Lib_Handler]::new()

$handler.dir = $General_Purpose_Libs
$handler.copy("Nexus")
$handler.copy("Socket")
$handler.copy("OS")
$handler.copy("SYS")
$handler.copy("Iterator")
$handler.copy("Timer")
$handler.copy("Date")
$handler.copy("CudaBridge")

$handler.dir = $eXtended_STL
$handler.copy("xvector")
$handler.copy("xstring")
$handler.copy("xmap")

$handler.dir = $Functional_STL
$handler.copy("re")
$handler.copy("ac")
$handler.copy("mc")

$handler.dir = $Modded_Objects
$handler.copy("cc")
$handler.copy("re2")


Write-Host "All Lib Folders Updated!!"
Write-Host "----------------------------------------------------------------------`n"
