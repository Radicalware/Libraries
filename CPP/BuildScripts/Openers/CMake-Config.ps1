#!/usr/bin/env pwsh

subl --new-window

Set-Location $PSScriptRoot

$(Get-ChildItem -Path ..\CMake\Config\* -Recurse).foreach({
    Write-Host $_.FullName;
    subl $_.FullName;
});
