#!/usr/bin/env pwsh

param (
    [switch] $Debug,
    [switch] $Clean, 
    [switch] $No_Exec,
    [switch] $Exec
)

# -----------------------------------
$proj_name  = "ex_xmap";
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

if($Exec){
	$argStruct = [Arg_Struct]::new($proj_name, [bool[]]($executable, $Debug, $Clean, $true));
	[Run_CMake]::new($argStruct).execute();
}else{
	$argStruct = [Arg_Struct]::new($proj_name, [bool[]]($executable, $Debug, $Clean, $true));
	$run = [Run_CMake]::new($argStruct).Print_Config().Link_n_Compile();

	if(!$No_Exec){ $run.execute(); }
}
