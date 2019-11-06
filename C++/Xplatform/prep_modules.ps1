#!/usr/bin/env pwsh

# All Class modules must be in "ps1" files, not "psm1" files.

$module_path = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules"
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules"
}

if(!$(Test-Path $module_path)){
    mkdir $module_path
}

Get-Module | Remove-Module

Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Modules\PS_Builder.ps1   $module_path -Force
Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Modules\Run_CMake.ps1    $module_path -Force
Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Modules\Arg_Struct.ps1   $module_path -Force

Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Manage\copy_projects.ps1 $module_path -Force
Copy-Item $PSScriptRoot\Unix_n_Windows_Scripts\Manage\copy_finds.ps1    $module_path -Force

