#!/usr/bin/env pwsh


Set-Location "$PSScriptRoot"

#&".\clean.ps1"


foreach($script in [string[]]("install.ps1","run.ps1")){
	$(Get-ChildItem -Path ../../ -Filter $script -Recurse).foreach({
		Write-Host "Installing: " $_.FullName
		&"$($_.FullName)" -Overwrite -No_Exec
	});
};

Write-Host "`n`nAll Libs Installed!!"

