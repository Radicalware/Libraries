#!/usr/bin/env pwsh

param (
    [switch] $Debug,
    [switch] $Clean, 
    [switch] $Overwrite,
    [switch] $BuildAllProjects,

    [switch] $NoMake,
    [switch] $NoCmake,
    [switch] $NoInstall,

    [switch] $Exec,  # only execute
    [switch] $NoExec # don't execute
)

# -----------------------------------
$proj_name  = "ex_Stats";
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


$ArgStruct = [Arg_Struct]::new($proj_name, $PSScriptRoot, [bool[]](
    $executable, $Debug, $Clean, $Overwrite, $BuildAllProjects, $noCmake, $noMake, $noInstall));

if($Exec){
    [Run_CMake]::new($ArgStruct).execute();
}else{
    $run = [Run_CMake]::new($ArgStruct).Print_Config().Link_n_Compile();
    if((!$NoMake) -and (!$NoExec)){ $run.execute(); }
}
