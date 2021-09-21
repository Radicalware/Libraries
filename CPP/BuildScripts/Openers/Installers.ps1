#!/usr/bin/env pwsh

subl --new-window

Set-Location $PSScriptRoot

$(Get-ChildItem -Path ..\..\ -Filter install.ps1 -Recurse).foreach({
    $name = $_.FullName
    Write-Host $name;
    subl $name;
});
