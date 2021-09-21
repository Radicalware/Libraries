#!/usr/bin/env pwsh

subl --new-window

Set-Location $PSScriptRoot

$(Get-ChildItem -Path ..\..\ -File -Recurse).foreach({
    $name = $_.FullName
    if($name -match "^.*Find.+\.cmake$"){
        Write-Host $name;
        subl $name;
    }
});
