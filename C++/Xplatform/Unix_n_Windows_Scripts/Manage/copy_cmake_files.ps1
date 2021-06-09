#!/usr/bin/env pwsh

Write-Host
Write-Host '-------------------------------------------------------------------------'

# Copies Cmake Files from "Source/Libraries" to "Souce/Cmake"

Write-Host "Updating Find Files: "

$ps1_path = $(Split-Path $MyInvocation.MyCommand.Path -Parent)

$find_cmake_path = ""
$rex = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $find_cmake_path = "/opt/Radicalware/CMake_Modules"
    $rex = "^.*/(Find[\w\d_-]+\.cmake)$"
}else{
    $find_cmake_path = "C:\source\CMake\Modules"
    $rex = "^.*\\(Find[\w\d_-]+\.cmake)$"
}

$(Get-ChildItem -Path $ps1_path/../../ -Recurse -Force).foreach({ 
	# -Force to find hidden files                
    $PathName = $_.FullName                                                                         
    if($PathName -match $rex){                    
        Write-Host $PathName;      
        Copy-Item $PathName $find_cmake_path -Force
    }                                                                                           
});                                                                                             


# Copy-Item "..\CMake\Config\*" $find_cmake_path

# Write-Host "All Find Files Updated!!"
# Write-Host '-------------------------------------------------------------------------'
# Write-Host

