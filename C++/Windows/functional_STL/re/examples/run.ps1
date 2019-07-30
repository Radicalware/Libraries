
param (
    [switch]$modify, # build_type (false = clean it / true = keep it)
    [switch]$debug   # false = Release
)

$exe_name = 'ex_re'

Set-Location $(Split-Path -parent $PSCommandPath)

$build_type = "Release"
if($debug){
    $build_type = "Debug"
}
$clean_build = ![switch]$modify;

Write-Host Options
Write-Host ==================================
Write-Host "Debug Mode Build Type = " $debug
Write-Host
Write-Host "Only Modify the Build = " $modify
Write-Host ==================================


..\..\..\cmake_installer.ps1 -project $exe_name -clean_build $clean_build -build_type $build_type
