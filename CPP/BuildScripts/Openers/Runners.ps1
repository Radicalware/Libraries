#!/usr/bin/env pwsh

subl --new-window
Start-Sleep -Milliseconds 200;

Set-Location $PSScriptRoot

$(Get-ChildItem -Path ..\..\ -Filter run.ps1 -Recurse).foreach({
    $name = $_.FullName
    Write-Host $name;
    subl $name;
    Start-Sleep -Milliseconds 100;
});
