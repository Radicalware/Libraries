#!/usr/bin/env pwsh

param (
    [switch] $Debug,
    [switch] $Clean, 
    [switch] $Overwrite,

    [switch] $NoCmake,
    [switch] $NoMake,
    [switch] $NoInstall
)


# -----------------------------------
$proj_name  = "Date";
$executable = $false;
# -----------------------------------

$module_path = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules"
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules"
}
Import-Module "$module_path\Arg_Struct.ps1" -Force
Import-Module "$module_path\Run_CMake.ps1" -Force
Set-Location $(Split-Path -parent $PSCommandPath)

$ArgStruct = [Arg_Struct]::new($proj_name, $PSScriptRoot, [bool[]]($executable, $Debug, $Clean, $Overwrite, $NoCmake, $NoMake, $NoInstall));
$run = [Run_CMake]::new($ArgStruct).Print_Config().Link_n_Compile();