#!/usr/bin/env pwsh

param (
    [switch] $Debug,
    [switch] $Clean, 
    [switch] $Overwrite,

    [switch] $NoMake,
    [switch] $NoCmake,
    [switch] $NoInstall,

    [switch] $Exec,  # only execute
    [switch] $NoExec # don't execute
)

# -----------------------------------
$proj_name  = "ex_SYS";
$executable = $true;
# -----------------------------------

$module_path = "";
if($($global:PSVersionTable.Platform -eq "Unix")){
    $module_path = "~/.local/share/powershell/Modules";
}else{
    $module_path = "$HOME\Documents\WindowsPowerShell\Modules";
}
Import-Module "$module_path\Arg_Struct.ps1" -Force;
Import-Module "$module_path\Run_CMake.ps1" -Force;
Set-Location $(Split-Path -parent $PSCommandPath);


if($Exec -and !$Debug){
    &"./execute.ps1"; 
}else{
    $ArgStruct = [Arg_Struct]::new($proj_name, $PSScriptRoot, [bool[]]($executable, $Debug, $Clean, $Overwrite, $noCmake, $noMake, $noInstall));
    $run = [Run_CMake]::new($argStruct).Print_Config().Link_n_Compile();

    if(!$No_Exec){ &"./execute.ps1"; }
}

