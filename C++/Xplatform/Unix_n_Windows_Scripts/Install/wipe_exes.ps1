#!/usr/bin/env pwsh

Set-Location $PSScriptRoot


#                     I hardcoded path for security reasons
$(Get-ChildItem -Path C:\source\include -Recurse -Force).foreach({ 
    # -Force to find hidden files
    $name = $_.FullName    
    if($_.PSIsContainer -and $name -match "^.*\\Release\\Release$"){
        if(Test-Path -Path $name){
            Write-Host $name;
            Remove-Item $_.FullName -Recurse -Force;
        }
    }
});
