#!/usr/bin/env pwsh


Set-Location "$PSScriptRoot"

#&".\clean.ps1"

Workflow Install_CMakes {
	param([System.Collections.ArrayList] $installs);

	foreach -parallel($install in $installs){
		InlineScript {   }
		InlineScript { "`n`n$($Using:install.FullName)`n"; &"$($Using:install.FullName)" -Overwrite -No_Exec };

	}
}

Install_CMakes $(Get-ChildItem -Path ..\..\ -Filter install.ps1 -Recurse);
Install_CMakes $(Get-ChildItem -Path ..\..\ -Filter run.ps1 -Recurse);

Get-Job | Receive-Job

Write-Host "`n`nAll Libs Installed!!"
