#!/usr/bin/env pwsh

subl --new-window
Start-Sleep -Milliseconds 100;

Set-Location $PSScriptRoot

$(Get-ChildItem -Path ..\CMake\Config\* -Recurse).foreach({
    Write-Host $_.FullName;
    subl $_.FullName;
    Start-Sleep -Milliseconds 100;
});
