#!/usr/bin/env pwsh

param (
    [switch] $modify,   # you shouldn't use this, look at code if you really want to,
    [switch] $lib,      # only install libs
    [switch] $Examples  # only install Examples
)

Set-Location "$PSScriptRoot"

$lib_installs = @()
$run_installs = @()

# ---------------------------------------------------------------------------------------
$lib_installs = @(
    "../../functional_STL/ac/install.ps1",
    "../../functional_STL/mc/install.ps1",
    "../../functional_STL/re/install.ps1",

    "../../General_Purpose_Libs/Nexus/install.ps1",

    "../../eXtended_STL/xstring/install.ps1",
    "../../eXtended_STL/xvector/install.ps1",
    "../../eXtended_STL/xmap/install.ps1",

    "../../General_Purpose_Libs/Iterator/install.ps1",
    "../../General_Purpose_Libs/OS/install.ps1",
    "../../General_Purpose_Libs/SYS/install.ps1",
    "../../General_Purpose_Libs/Timer/install.ps1",

    "../../Modded_Objects/cc/install.ps1"
)
    # --------------------------------------------------
$run_installs = @(
    "../../functional_STL/ac/Examples/run.ps1",
    "../../functional_STL/mc/Examples/run.ps1",
    "../../functional_STL/re/Examples/run.ps1",

    "../../General_Purpose_Libs/Nexus/Examples/run.ps1",

    "../../eXtended_STL/xvector/Examples/run.ps1",
    "../../eXtended_STL/xstring/example/run.ps1",
    "../../eXtended_STL/xmap/Examples/run.ps1",

    "../../General_Purpose_Libs/OS/Examples/run.ps1",
    "../../General_Purpose_Libs/SYS/Examples/run.ps1",
    "../../General_Purpose_Libs/Timer/Examples/run.ps1",

    "../../Modded_Objects/cc/Examples/run.ps1"
)

# ---------------------------------------------------------------------------------------

if($lib -eq $false -and $Examples -eq $false){
    $lib = $true;
    $Examples = $true;
}

if($lib -eq $true){
    foreach($install in $lib_installs){
        Set-Location "$PSScriptRoot"

        Write-Host $install
        &"$install" -Overwrite -No_Exec
    }
}
#if($Examples -eq $true){
#    foreach($install in $run_installs){
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
#    Install_CMakes $($lib_installs)
#}

if($Examples -eq $true){
    Install_CMakes $(Get-ChildItem -Path ..\..\ -Filter run.ps1 -Recurse);
    #Install_CMakes $run_installs
}

Get-Job | Receive-Job

# ---------------------------------------------------------------------------------------

Write-Host "`n`nAll Libs Installed!!"
Set-Location "$PSScriptRoot"
