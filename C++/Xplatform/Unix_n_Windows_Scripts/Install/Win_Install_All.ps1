#!/usr/bin/env pwsh

using module ".\files.psm1"

param (
    [switch] $modify,   # you shouldn't use this, look at code if you really want to,
    [switch] $lib,      # only install libs
    [switch] $Examples  # only install Examples
)

Set-Location "$PSScriptRoot"
$files = [Files]::new()

# ---------------------------------------------------------------------------------------

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
#if($Examples -eq $true){
#    foreach($install in $files.examples){
#        Set-Location "$PSScriptRoot"
#
#        Write-Host $install
#        &"$install" -Overwrite -No_Exec
#    }
#}
    

# ---------------------------------------------------------------------------------------
    

Workflow Install_CMakes {
    param([System.Collections.ArrayList] $installs);

    foreach -parallel($install in $installs){
        InlineScript {   }
        InlineScript { "`n`n$($Using:install.FullName)`n"; &"$($Using:install.FullName)" -Overwrite -No_Exec };

    }
}


#if($lib -eq $true){
#    #Install_CMakes $(Get-ChildItem -Path ..\..\ -Filter install.ps1 -Recurse);
#    Install_CMakes $($files.libs)
#}

if($Examples -eq $true){
    Install_CMakes $(Get-ChildItem -Path ..\..\ -Filter run.ps1 -Recurse);
    #Install_CMakes $files.examples
}

Get-Job | Receive-Job

# ---------------------------------------------------------------------------------------
    

Write-Host "`n`nAll Libs Installed!!"
Set-Location "$PSScriptRoot"
