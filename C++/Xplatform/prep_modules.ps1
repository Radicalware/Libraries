#!/usr/bin/env pwsh

# All Class modules must be in "ps1" files, not "psm1" files.

$module_path = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules"
    $config_path = "/opt/Radicalware/CMake_Modules"
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules"
    $config_path = "C:\Source\CMake\Modules"
}

if(!$(Test-Path $module_path)){
    mkdir $module_path
}

Get-Module | Remove-Module

Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Modules\* $module_path -Force
Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Manage\*  $module_path -Force

Copy-Item ./Unix_n_Windows_Scripts/CMake/Config/*  $config_path -Force
Copy-Item ./Unix_n_Windows_Scripts/CMake/Modules/* $config_path -Force
