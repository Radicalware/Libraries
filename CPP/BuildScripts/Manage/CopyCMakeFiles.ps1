#!/usr/bin/env pwsh

Write-Host
Write-Host '--------------------------------------------------------------------------------------------'

Write-Host "Updating Find Files: "
Write-Host

$FindCMakePath = ""
$rex = ""
if($($global:PSVersionTable.Platform -eq "Unix")){
    $FindCMakePath = "/opt/Radicalware/CMake_Modules"
    $rex = "^.*/(Find[\w\d_-]+\.cmake|Radical[-\w\d]+\.cmake)$"
}else{
    $FindCMakePath = "C:\source\CMake\Modules"
    $rex = "^.*\\(Find[\w\d_-]+\.cmake|Radical[-\w\d]+\.cmake)$"
}

$(Get-ChildItem -Path $PSScriptRoot/../../ -Recurse -Force).foreach({ 
	# -Force to find hidden files                
    $PathName = $_.FullName                                                                         
    if($PathName -match $rex){                    
        Write-Host $PathName;      
        Copy-Item $PathName $FindCMakePath -Force
    }                                                                                           
});                                                                                             
Write-Host
Write-Host "All CMake Files Updated!!"
Write-Host '--------------------------------------------------------------------------------------------'
Write-Host

