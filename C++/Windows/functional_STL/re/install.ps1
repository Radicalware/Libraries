
param (
    [switch]$modify, # build_type (false = clean it / true = keep it)
    [switch]$debug   # false = Release
)

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
Write-Host "Clear All Files       = " $modify
Write-Host ==================================


..\..\cmake_installer.ps1 -project re -clean_build $clean_build -build_type $build_type
