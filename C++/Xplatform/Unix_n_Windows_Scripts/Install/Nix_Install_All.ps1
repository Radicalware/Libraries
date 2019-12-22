#!/usr/bin/env pwsh

using module "./files.psm1"

param (
    [switch] $modify,   # you shouldn't use this, look at code if you really want to
    [switch] $lib,      # only install libs
    [switch] $Examples  # only install Examples
)

Set-Location "$PSScriptRoot"

#&".\clean.ps1"

# If you add a new lib, and only need to modify you can use the following code
# It is highly advised not to in most situations, you want to install
# dependencies in order.

$files = [Files]::new()

if($modify){
    foreach($script in [string[]]("install.ps1","run.ps1")){
        $(Get-ChildItem -Path ../../ -Filter $script -Recurse).foreach({
            Write-Host "Installing: " $_.FullName
            &"$($_.FullName)" -Overwrite -No_Exec
        });
    };
}else{
    if($lib -eq $false -and $Examples -eq $false){
        $lib = $true;
        $Examples = $true;
    }

    if($lib -eq $true){
        foreach($install in $files.libs){
            Set-Location "$PSScriptRoot"

            Write-Host $install
            &"$install" -Overwrite -No_Exec
        }
    }
    if($Examples -eq $true){
        foreach($install in $files.examples){
            Set-Location "$PSScriptRoot"

            Write-Host $install
            &"$install" -Overwrite -No_Exec
        }
    }
}


Write-Host "`n`nAll Libs Installed!!"
