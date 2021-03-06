#!/usr/bin/env pwsh

Write-Host
Write-Host '-------------------------------------------------------------------------'

Write-Host "Updating Find Files: "

$find_cmake_path = ""
$rex = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $find_cmake_path = "/opt/Radicalware/CMake_Modules"
    $rex = "^.*/(Find[\w\d_-]+\.cmake)$"
}else{
    $rex = "^.*\\(Find[\w\d_-]+\.cmake)$"
    $find_cmake_path = "C:\source\CMake\Modules"
}

$(Get-ChildItem -Path ../../ -Recurse -Force).foreach({ 
	# -Force to find hidden files                
    $name = $_.FullName                                                                         
    if($name -match $rex){                    
        if(Test-Path -Path $name){                                                              
            Write-Host $name;      
            Copy-Item $name $find_cmake_path -Force
        }                                                                                       
    }                                                                                           
});                                                                                             


Write-Host "All Find Files Updated!!"
Write-Host '-------------------------------------------------------------------------'
Write-Host

