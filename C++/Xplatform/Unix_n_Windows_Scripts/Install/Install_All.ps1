#!/usr/bin/env pwsh

using module "./files.psm1"

param (
    [switch] $lib,      # Only install libs
    [switch] $Examples, # Only install Examples
    [switch] $Debug,    # Install Debug instead of Release
    [string] $MatchOnly # Install only what matches this pattern
)

$current_location = "$PSScriptRoot"
Set-Location $current_location

#&".\clean.ps1"

# If you add a new lib, and only need to modify you can use the following code
# It is highly advised not to in most situations, you want to install
# dependencies in order.

$RegexStr = "(?i)" + $MatchOnly
$RegexPattern = [Regex]::new($RegexStr)

$files = [Files]::new()

if($lib -eq $false -and $Examples -eq $false){
    $lib = $true;
    $Examples = $true;
}

if($lib -eq $true){
    foreach($install in $files.libs){

        if($MatchOnly.Length -gt 0 -and $RegexPattern.Match($install).Success -ne $true)
        {
            continue;
        }

        Set-Location "$PSScriptRoot"

        Write-Host $install
        if($Debug -eq $true){
            &"$install" -Overwrite -No_Exec -Debug
        }else{
            &"$install" -Overwrite -No_Exec
        }
    }
}
if($Examples -eq $true){
    foreach($install in $files.examples){

        if($MatchOnly.Length -gt 0 -and $RegexPattern.Match($install).Success -ne $true)
        {
            continue;
        }

        Set-Location "$PSScriptRoot"

        Write-Host $install
        if($Debug -eq $true){
            &"$install" -Overwrite -No_Exec -Debug
        }else{
            &"$install" -Overwrite -No_Exec
        }
    }
}


Write-Host "`n`nAll Libs Installed!!"

cd $current_location