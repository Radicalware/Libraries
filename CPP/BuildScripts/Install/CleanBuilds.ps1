#!/usr/bin/env pwsh

param (
    [switch] $CleanInstalled
)

Set-Location $PSScriptRoot

if($CleanInstalled -eq $true){
    Write-Host "Deep Clean"
    Remove-Item -Force -Recurse C:\Source\CMake\Radicalware\Libraries
}
else{
    Write-Host "Shallow Clean"
}


#                     I hardcoded path for security reasons
$(Get-ChildItem -Path C:\Source\Radicalware\Libraries -Recurse -Force).foreach({ 
    # -Force to find hidden files
    $name = $_.FullName    
    if($_.PSIsContainer -and $name -match "^.*\\(Release|Debug|Build|out|\.vs|.vscode|cmake-build-debug|.idea)$"){
        if(Test-Path -Path $name){
            Write-Host $name;
            Remove-Item $_.FullName -Recurse -Force;
        }
    }
});
