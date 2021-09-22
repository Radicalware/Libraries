#!/usr/bin/env pwsh

# All Class modules must be in "ps1" files, not "psm1" files.

param (
    [switch] $LinkDirs
)

$PowershellModulePath = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $PowershellModulePath = "~/.local/share/powershell/Modules"
    $CMakeModulePath = "/opt/Radicalware/CMake_Modules"
}else{
    $PowershellModulePath = "$HOME\Documents\WindowsPowerShell\Modules"
    $CMakeModulePath = "C:\Source\CMake\Modules"

    if($(Test-Path $CMakeModulePath) -eq $false)
    {
        mkdir C:\Source
        mkdir C:\Source\CMake
        mkdir C:\Source\CMake\Modules
    }
}

if($LinkDirs) # To samve runtime and clutter, I decided not to check every path but instead use a cmd line switch
{
    mkdir C:\Source\CMake\Radicalware\Libraries\Build\Release\lib
    mkdir C:\Source\CMake\Radicalware\Libraries\Build\Release\bin

    mkdir C:\Source\CMake\Radicalware\Libraries\Build\Debug\lib
    mkdir C:\Source\CMake\Radicalware\Libraries\Build\Debug\bin

    mkdir C:\Source\CMake\Radicalware\Applications\Build\Release
    mkdir C:\Source\CMake\Radicalware\Applications\Build\Debug

    cmd /c mklink /D C:\Source\CMake\Radicalware\Applications\Build\Release\lib C:\Source\CMake\Radicalware\Libraries\Build\Release\lib
    cmd /c mklink /D C:\Source\CMake\Radicalware\Applications\Build\Release\bin C:\Source\CMake\Radicalware\Libraries\Build\Release\bin

    cmd /c mklink /D C:\Source\CMake\Radicalware\Applications\Build\Debug\lib C:\Source\CMake\Radicalware\Libraries\Build\Debug\lib
    cmd /c mklink /D C:\Source\CMake\Radicalware\Applications\Build\Debug\bin C:\Source\CMake\Radicalware\Libraries\Build\Debug\bin
}

Get-Module | Remove-Module

Write-Host
Write-Host "Copying Powershell Modules"
Copy-Item $PSScriptRoot\BuildScripts\Modules\* $PowershellModulePath -Force
Copy-Item $PSScriptRoot\BuildScripts\Manage\*  $PowershellModulePath -Force

&"$PSScriptRoot\BuildScripts\Manage\CopyCMakeFiles.ps1"


if($LinkDirs -eq $false){
    Write-Host "WARNING: Directory Linking Skipped!!" -ForegroundColor Yellow
    Write-Host "You can run with the -LinkDirs option to link them`n"
}
